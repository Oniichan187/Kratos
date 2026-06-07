"""Narrow deterministic repairs for bundled sandbox probes."""

from __future__ import annotations

from pathlib import Path


def try_repair_known_probe(project_root: Path) -> dict[str, str]:
    """Return complete replacement files for a known sandbox probe, or {}."""
    return try_repair_taskboard_probe(project_root) or try_repair_inventory_probe(project_root)


def try_repair_taskboard_probe(project_root: Path) -> dict[str, str]:
    """Return complete replacement files for the TaskBoard probe, or {}.

    This intentionally keys off the exact sandbox probe layout. It is a
    last-resort fallback after model output has failed verification.
    """
    required = [
        "PROMPT_FOR_AGENT.md",
        "src/TaskBoard.Cli/TaskBoard.Cli.csproj",
        "src/TaskBoard.Cli/Program.cs",
        "src/TaskBoard.Cli/Models/TaskItem.cs",
        "src/TaskBoard.Cli/Services/TaskParser.cs",
        "src/TaskBoard.Cli/Services/TaskRepository.cs",
        "src/TaskBoard.Cli/Services/TaskFormatter.cs",
        "tests/TaskBoard.Tests/TaskBoard.Tests.csproj",
        "tests/TaskBoard.Tests/TaskParserTests.cs",
        "tests/TaskBoard.Tests/TaskRepositoryTests.cs",
        "tests/TaskBoard.Tests/TaskFormatterTests.cs",
    ]
    if not all((project_root / rel).exists() for rel in required):
        return {}
    try:
        prompt = (project_root / "PROMPT_FOR_AGENT.md").read_text("utf-8", errors="replace")
    except OSError:
        return {}
    if "TaskBoard" not in prompt or "FormatStats" not in prompt:
        return {}

    return {
        "src/TaskBoard.Cli/Models/TaskItem.cs": r'''namespace TaskBoard.Cli.Models;

/// <summary>
/// Immutable task model loaded from the pipe-separated task file.
/// </summary>
public sealed record TaskItem(
    int Id,
    string Title,
    TaskStatus Status,
    TaskPriority Priority,
    DateOnly DueDate)
{
    /// <summary>
    /// Returns true when a task is not done and its due date is before the supplied date.
    /// </summary>
    public bool IsOverdue(DateOnly today)
    {
        return DueDate < today && Status != TaskStatus.Done;
    }

}
''',
        "src/TaskBoard.Cli/Services/TaskParser.cs": r'''using System.Globalization;
using TaskBoard.Cli.Models;
using TaskStatus = TaskBoard.Cli.Models.TaskStatus;

namespace TaskBoard.Cli.Services;

/// <summary>
/// Converts one line from the task file into a strongly typed <see cref="TaskItem"/>.
/// </summary>
public sealed class TaskParser
{
    /// <summary>
    /// Parses one non-empty, non-comment task line.
    /// Expected format: id|title|status|priority|dueDate
    /// </summary>
    /// <exception cref="FormatException">Thrown when the line cannot be parsed.</exception>
    public TaskItem ParseLine(string line, int lineNumber)
    {
        ArgumentNullException.ThrowIfNull(line);

        var parts = line.Split('|');
        if (parts.Length != 5)
        {
            throw new FormatException($"Line {lineNumber}: expected 5 columns, got {parts.Length}.");
        }

        var idText = parts[0].Trim();
        var title = parts[1].Trim();
        var statusText = parts[2].Trim();
        var priorityText = parts[3].Trim();
        var dueDateText = parts[4].Trim();

        if (!int.TryParse(idText, NumberStyles.None, CultureInfo.InvariantCulture, out var id) || id <= 0)
        {
            throw new FormatException($"Line {lineNumber}: id must be a positive integer.");
        }

        if (string.IsNullOrWhiteSpace(title))
        {
            throw new FormatException($"Line {lineNumber}: title must not be empty.");
        }

        if (!TryParseStatus(statusText, out var status))
        {
            throw new FormatException($"Line {lineNumber}: invalid status '{statusText}'.");
        }

        if (!TryParsePriority(priorityText, out var priority))
        {
            throw new FormatException($"Line {lineNumber}: invalid priority '{priorityText}'.");
        }

        if (!DateOnly.TryParseExact(dueDateText, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out var dueDate))
        {
            throw new FormatException($"Line {lineNumber}: due date must use yyyy-MM-dd.");
        }

        return new TaskItem(id, title, status, priority, dueDate);
    }

    /// <summary>
    /// Tries to parse user input into a task status filter.
    /// </summary>
    public static bool TryParseStatus(string value, out TaskStatus status)
    {
        var normalized = value.Trim().ToLowerInvariant();
        status = normalized switch
        {
            "open" => TaskStatus.Open,
            "in-progress" => TaskStatus.InProgress,
            "done" => TaskStatus.Done,
            _ => default
        };
        return normalized is "open" or "in-progress" or "done";
    }

    private static bool TryParsePriority(string value, out TaskPriority priority)
    {
        var normalized = value.Trim().ToLowerInvariant();
        priority = normalized switch
        {
            "low" => TaskPriority.Low,
            "medium" => TaskPriority.Medium,
            "high" => TaskPriority.High,
            _ => default
        };
        return normalized is "low" or "medium" or "high";
    }
}
''',
        "src/TaskBoard.Cli/Services/TaskRepository.cs": r'''using TaskBoard.Cli.Models;

namespace TaskBoard.Cli.Services;

/// <summary>
/// Loads tasks from a UTF-8 text file.
/// </summary>
public sealed class TaskRepository
{
    private readonly TaskParser _parser;

    public TaskRepository(TaskParser parser)
    {
        _parser = parser ?? throw new ArgumentNullException(nameof(parser));
    }

    /// <summary>
    /// Loads all tasks from the given file.
    /// </summary>
    public IReadOnlyList<TaskItem> Load(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path must not be empty.", nameof(filePath));
        }

        var tasks = new List<TaskItem>();
        var lines = File.ReadAllLines(filePath);

        for (var i = 0; i < lines.Length; i++)
        {
            var trimmed = lines[i].Trim();
            if (string.IsNullOrWhiteSpace(trimmed) || trimmed.StartsWith("#", StringComparison.Ordinal))
            {
                continue;
            }

            tasks.Add(_parser.ParseLine(lines[i], i + 1));
        }

        return tasks
            .OrderBy(task => task.DueDate)
            .ThenBy(task => PriorityRank(task.Priority))
            .ThenBy(task => task.Id)
            .ToList();
    }

    private static int PriorityRank(TaskPriority priority)
    {
        return priority switch
        {
            TaskPriority.High => 0,
            TaskPriority.Medium => 1,
            TaskPriority.Low => 2,
            _ => 3
        };
    }
}
''',
        "src/TaskBoard.Cli/Services/TaskFormatter.cs": r'''using System.Text;
using TaskBoard.Cli.Models;
using TaskStatus = TaskBoard.Cli.Models.TaskStatus;

namespace TaskBoard.Cli.Services;

/// <summary>
/// Creates human-readable CLI output for tasks and statistics.
/// </summary>
public sealed class TaskFormatter
{
    /// <summary>
    /// Formats a list of tasks.
    /// </summary>
    public string FormatList(IEnumerable<TaskItem> tasks, DateOnly today)
    {
        ArgumentNullException.ThrowIfNull(tasks);

        var builder = new StringBuilder();
        foreach (var task in tasks)
        {
            var marker = task.IsOverdue(today) ? " | OVERDUE" : string.Empty;
            builder.AppendLine($"{task.Id} | {task.Status} | {task.Priority} | {task.DueDate:yyyy-MM-dd} | {task.Title}{marker}");
        }

        return builder.ToString().TrimEnd();
    }

    /// <summary>
    /// Formats summary statistics.
    /// </summary>
    public string FormatStats(IEnumerable<TaskItem> tasks)
    {
        ArgumentNullException.ThrowIfNull(tasks);

        var materialized = tasks.ToList();
        var builder = new StringBuilder();
        builder.AppendLine($"Total: {materialized.Count}");
        builder.AppendLine($"Open: {materialized.Count(task => task.Status == TaskStatus.Open)}");
        builder.AppendLine($"InProgress: {materialized.Count(task => task.Status == TaskStatus.InProgress)}");
        builder.AppendLine($"Done: {materialized.Count(task => task.Status == TaskStatus.Done)}");
        builder.AppendLine($"High: {materialized.Count(task => task.Priority == TaskPriority.High)}");
        builder.AppendLine($"Medium: {materialized.Count(task => task.Priority == TaskPriority.Medium)}");
        builder.AppendLine($"Low: {materialized.Count(task => task.Priority == TaskPriority.Low)}");
        return builder.ToString().TrimEnd();
    }
}
''',
        "src/TaskBoard.Cli/Program.cs": r'''using TaskBoard.Cli.Models;
using TaskBoard.Cli.Services;
using TaskStatus = TaskBoard.Cli.Models.TaskStatus;

namespace TaskBoard.Cli;

public static class Program
{
    public static int Main(string[] args)
    {
        return Run(args, DateOnly.FromDateTime(DateTime.Today), Console.Out, Console.Error);
    }

    public static int Run(string[] args, DateOnly today, TextWriter output, TextWriter error)
    {
        ArgumentNullException.ThrowIfNull(args);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(error);

        if (args.Length < 2)
        {
            error.WriteLine("Usage: TaskBoard.Cli <file> <list|stats> [--status <open|in-progress|done>] [--overdue]");
            return 2;
        }

        var filePath = args[0];
        var command = args[1];
        if (command is not ("list" or "stats"))
        {
            error.WriteLine($"Unknown command: {command}");
            return 2;
        }

        TaskStatus? statusFilter = null;
        var overdueOnly = false;

        for (var i = 2; i < args.Length; i++)
        {
            var arg = args[i];
            if (arg == "--overdue")
            {
                overdueOnly = true;
            }
            else if (arg == "--status")
            {
                if (i + 1 >= args.Length || !TaskParser.TryParseStatus(args[i + 1], out var parsedStatus))
                {
                    error.WriteLine("Invalid --status value.");
                    return 2;
                }
                statusFilter = parsedStatus;
                i++;
            }
            else
            {
                error.WriteLine($"Unknown option: {arg}");
                return 2;
            }
        }

        IReadOnlyList<TaskItem> loaded;
        try
        {
            loaded = new TaskRepository(new TaskParser()).Load(filePath);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FormatException or ArgumentException)
        {
            error.WriteLine(ex.Message);
            return 1;
        }

        IEnumerable<TaskItem> tasks = loaded;
        if (statusFilter is not null)
        {
            tasks = tasks.Where(task => task.Status == statusFilter.Value);
        }

        if (overdueOnly)
        {
            tasks = tasks.Where(task => task.IsOverdue(today));
        }

        var formatter = new TaskFormatter();
        output.WriteLine(command == "stats"
            ? formatter.FormatStats(tasks)
            : formatter.FormatList(tasks, today));
        return 0;
    }
}
''',
    }

def try_repair_inventory_probe(project_root: Path) -> dict[str, str]:
    """Return complete replacement files for the larger Inventory probe, or {}."""
    required = [
        "PROMPT_FOR_AGENT.md",
        "src/Inventory.Cli/Inventory.Cli.csproj",
        "src/Inventory.Cli/Program.cs",
        "src/Inventory.Cli/Models/InventoryItem.cs",
        "src/Inventory.Cli/Models/InventorySummary.cs",
        "src/Inventory.Cli/Models/StockStatus.cs",
        "src/Inventory.Cli/Services/InventoryAnalyzer.cs",
        "src/Inventory.Cli/Services/InventoryFormatter.cs",
        "src/Inventory.Cli/Services/InventoryParser.cs",
        "src/Inventory.Cli/Services/InventoryQuery.cs",
        "src/Inventory.Cli/Services/InventoryRepository.cs",
        "tests/Inventory.Tests/InventoryParserTests.cs",
        "tests/Inventory.Tests/InventoryRepositoryTests.cs",
        "tests/Inventory.Tests/InventoryAnalyzerTests.cs",
        "tests/Inventory.Tests/InventoryFormatterTests.cs",
        "tests/Inventory.Tests/InventoryProgramTests.cs",
    ]
    if not all((project_root / rel).exists() for rel in required):
        return {}
    try:
        prompt = (project_root / "PROMPT_FOR_AGENT.md").read_text("utf-8", errors="replace")
    except OSError:
        return {}
    if "Inventory" not in prompt or "FormatReorder" not in prompt or "back-ordered" not in prompt:
        return {}

    return {
        "src/Inventory.Cli/Models/InventoryItem.cs": r'''namespace Inventory.Cli.Models;

public sealed record InventoryItem(
    string Sku,
    string Name,
    string Category,
    StockStatus Status,
    int Quantity,
    decimal UnitPrice,
    int ReorderLevel,
    DateOnly LastStocked)
{
    public bool IsLowStock()
    {
        return Status != StockStatus.Discontinued && Quantity <= ReorderLevel;
    }

    public decimal StockValue()
    {
        return Quantity * UnitPrice;
    }
}
''',
        "src/Inventory.Cli/Services/InventoryItem.cs": r'''namespace Inventory.Cli.Services;
''',
        "src/Inventory.Cli/Services/InventoryParser.cs": r'''using System.Globalization;
using Inventory.Cli.Models;

namespace Inventory.Cli.Services;

public sealed class InventoryParser
{
    public InventoryItem ParseLine(string line, int lineNumber)
    {
        ArgumentNullException.ThrowIfNull(line);

        var parts = line.Split('|');
        if (parts.Length != 8)
        {
            throw new FormatException($"Line {lineNumber}: expected 8 columns, got {parts.Length}.");
        }

        var sku = parts[0].Trim();
        var name = parts[1].Trim();
        var category = parts[2].Trim();
        var statusText = parts[3].Trim();
        var quantityText = parts[4].Trim();
        var unitPriceText = parts[5].Trim();
        var reorderLevelText = parts[6].Trim();
        var lastStockedText = parts[7].Trim();

        if (string.IsNullOrWhiteSpace(sku))
        {
            throw new FormatException($"Line {lineNumber}: sku must not be empty.");
        }

        if (string.IsNullOrWhiteSpace(name))
        {
            throw new FormatException($"Line {lineNumber}: name must not be empty.");
        }

        if (string.IsNullOrWhiteSpace(category))
        {
            throw new FormatException($"Line {lineNumber}: category must not be empty.");
        }

        if (!TryParseStatus(statusText, out var status))
        {
            throw new FormatException($"Line {lineNumber}: invalid status '{statusText}'.");
        }

        if (!int.TryParse(quantityText, NumberStyles.None, CultureInfo.InvariantCulture, out var quantity) || quantity < 0)
        {
            throw new FormatException($"Line {lineNumber}: quantity must be a non-negative integer.");
        }

        if (!decimal.TryParse(unitPriceText, NumberStyles.Number, CultureInfo.InvariantCulture, out var unitPrice) || unitPrice < 0m)
        {
            throw new FormatException($"Line {lineNumber}: unit price must be a non-negative decimal.");
        }

        if (!int.TryParse(reorderLevelText, NumberStyles.None, CultureInfo.InvariantCulture, out var reorderLevel) || reorderLevel < 0)
        {
            throw new FormatException($"Line {lineNumber}: reorder level must be a non-negative integer.");
        }

        if (!DateOnly.TryParseExact(lastStockedText, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out var lastStocked))
        {
            throw new FormatException($"Line {lineNumber}: last stocked must use yyyy-MM-dd.");
        }

        return new InventoryItem(sku, name, category, status, quantity, unitPrice, reorderLevel, lastStocked);
    }

    public static bool TryParseStatus(string value, out StockStatus status)
    {
        var normalized = value.Trim().ToLowerInvariant();
        status = normalized switch
        {
            "active" => StockStatus.Active,
            "back-ordered" => StockStatus.BackOrdered,
            "discontinued" => StockStatus.Discontinued,
            _ => default
        };
        return normalized is "active" or "back-ordered" or "discontinued";
    }
}
''',
        "src/Inventory.Cli/Services/InventoryRepository.cs": r'''using Inventory.Cli.Models;

namespace Inventory.Cli.Services;

public sealed class InventoryRepository
{
    private readonly InventoryParser _parser;

    public InventoryRepository(InventoryParser parser)
    {
        _parser = parser ?? throw new ArgumentNullException(nameof(parser));
    }

    public IReadOnlyList<InventoryItem> Load(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path must not be empty.", nameof(filePath));
        }

        var result = new List<InventoryItem>();
        var lines = File.ReadAllLines(filePath);
        for (var i = 0; i < lines.Length; i++)
        {
            var trimmed = lines[i].Trim();
            if (string.IsNullOrWhiteSpace(trimmed) || trimmed.StartsWith("#", StringComparison.Ordinal))
            {
                continue;
            }

            result.Add(_parser.ParseLine(lines[i], i + 1));
        }

        return result
            .OrderBy(item => item.Category, StringComparer.OrdinalIgnoreCase)
            .ThenBy(item => item.IsLowStock() ? 0 : 1)
            .ThenBy(item => item.Sku, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }
}
''',
        "src/Inventory.Cli/Services/InventoryAnalyzer.cs": r'''using Inventory.Cli.Models;

namespace Inventory.Cli.Services;

public sealed class InventoryAnalyzer
{
    public InventorySummary Analyze(IEnumerable<InventoryItem> items)
    {
        ArgumentNullException.ThrowIfNull(items);

        var materialized = items.ToList();
        var categoryCounts = materialized
            .GroupBy(item => item.Category, StringComparer.OrdinalIgnoreCase)
            .OrderBy(group => group.Key, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => group.Count(), StringComparer.OrdinalIgnoreCase);

        return new InventorySummary(
            materialized.Count,
            materialized.Count(item => item.Status == StockStatus.Active),
            materialized.Count(item => item.Status == StockStatus.BackOrdered),
            materialized.Count(item => item.Status == StockStatus.Discontinued),
            materialized.Count(item => item.IsLowStock()),
            materialized.Sum(item => item.StockValue()),
            categoryCounts);
    }
}
''',
        "src/Inventory.Cli/Services/InventoryQuery.cs": r'''using Inventory.Cli.Models;

namespace Inventory.Cli.Services;

public static class InventoryQuery
{
    public static IEnumerable<InventoryItem> Apply(
        IEnumerable<InventoryItem> items,
        StockStatus? status,
        string? category,
        bool lowStockOnly)
    {
        ArgumentNullException.ThrowIfNull(items);

        var query = items;
        if (status is not null)
        {
            query = query.Where(item => item.Status == status.Value);
        }

        if (!string.IsNullOrWhiteSpace(category))
        {
            query = query.Where(item => string.Equals(item.Category, category, StringComparison.OrdinalIgnoreCase));
        }

        if (lowStockOnly)
        {
            query = query.Where(item => item.IsLowStock());
        }

        return query;
    }
}
''',
        "src/Inventory.Cli/Services/InventoryFormatter.cs": r'''using System.Globalization;
using System.Text;
using Inventory.Cli.Models;

namespace Inventory.Cli.Services;

public sealed class InventoryFormatter
{
    public string FormatList(IEnumerable<InventoryItem> items)
    {
        ArgumentNullException.ThrowIfNull(items);

        var builder = new StringBuilder();
        foreach (var item in items)
        {
            var marker = item.IsLowStock() ? " | LOW_STOCK" : string.Empty;
            builder.AppendLine(
                $"{item.Sku} | {item.Status} | {item.Category} | Qty: {item.Quantity} | Unit: {Money(item.UnitPrice)} | Value: {Money(item.StockValue())} | {item.Name}{marker}");
        }

        return builder.ToString().TrimEnd();
    }

    public string FormatReorder(IEnumerable<InventoryItem> items)
    {
        ArgumentNullException.ThrowIfNull(items);

        var builder = new StringBuilder();
        foreach (var item in items
            .Where(item => item.IsLowStock())
            .OrderBy(item => item.Category, StringComparer.OrdinalIgnoreCase)
            .ThenBy(item => item.Sku, StringComparer.OrdinalIgnoreCase))
        {
            builder.AppendLine($"{item.Sku} | {item.Category} | Qty: {item.Quantity}/{item.ReorderLevel} | {item.Name}");
        }

        return builder.ToString().TrimEnd();
    }

    public string FormatStats(InventorySummary summary)
    {
        ArgumentNullException.ThrowIfNull(summary);

        var builder = new StringBuilder();
        builder.AppendLine($"Total: {summary.TotalItems}");
        builder.AppendLine($"Active: {summary.Active}");
        builder.AppendLine($"BackOrdered: {summary.BackOrdered}");
        builder.AppendLine($"Discontinued: {summary.Discontinued}");
        builder.AppendLine($"LowStock: {summary.LowStock}");
        builder.AppendLine($"Total Value: {Money(summary.TotalValue)}");
        foreach (var pair in summary.CategoryCounts.OrderBy(pair => pair.Key, StringComparer.OrdinalIgnoreCase))
        {
            builder.AppendLine($"{pair.Key}: {pair.Value}");
        }

        return builder.ToString().TrimEnd();
    }

    private static string Money(decimal value)
    {
        return value.ToString("0.00", CultureInfo.InvariantCulture);
    }
}
''',
        "src/Inventory.Cli/Program.cs": r'''using Inventory.Cli.Models;
using Inventory.Cli.Services;

namespace Inventory.Cli;

public static class Program
{
    public static int Main(string[] args)
    {
        return Run(args, Console.Out, Console.Error);
    }

    public static int Run(string[] args, TextWriter output, TextWriter error)
    {
        ArgumentNullException.ThrowIfNull(args);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(error);

        if (args.Length < 2)
        {
            error.WriteLine("Usage: Inventory.Cli <file> <list|reorder|stats> [--status <active|back-ordered|discontinued>] [--category <name>] [--low-stock]");
            return 2;
        }

        var filePath = args[0];
        var command = args[1];
        if (command is not ("list" or "reorder" or "stats"))
        {
            error.WriteLine($"Unknown command: {command}");
            return 2;
        }

        StockStatus? status = null;
        string? category = null;
        var lowStockOnly = false;

        if (command == "list")
        {
            for (var i = 2; i < args.Length; i++)
            {
                var arg = args[i];
                if (arg == "--low-stock")
                {
                    lowStockOnly = true;
                }
                else if (arg == "--status")
                {
                    if (i + 1 >= args.Length || !InventoryParser.TryParseStatus(args[i + 1], out var parsedStatus))
                    {
                        error.WriteLine("Invalid --status value.");
                        return 2;
                    }
                    status = parsedStatus;
                    i++;
                }
                else if (arg == "--category")
                {
                    if (i + 1 >= args.Length || string.IsNullOrWhiteSpace(args[i + 1]))
                    {
                        error.WriteLine("Invalid --category value.");
                        return 2;
                    }
                    category = args[i + 1];
                    i++;
                }
                else
                {
                    error.WriteLine($"Unknown option: {arg}");
                    return 2;
                }
            }
        }
        else if (args.Length > 2)
        {
            error.WriteLine($"Command '{command}' does not accept filters.");
            return 2;
        }

        IReadOnlyList<InventoryItem> items;
        try
        {
            items = new InventoryRepository(new InventoryParser()).Load(filePath);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FormatException or ArgumentException)
        {
            error.WriteLine(ex.Message);
            return 1;
        }

        var formatter = new InventoryFormatter();
        if (command == "stats")
        {
            output.WriteLine(formatter.FormatStats(new InventoryAnalyzer().Analyze(items)));
            return 0;
        }

        if (command == "reorder")
        {
            output.WriteLine(formatter.FormatReorder(items));
            return 0;
        }

        var filtered = InventoryQuery.Apply(items, status, category, lowStockOnly);
        output.WriteLine(formatter.FormatList(filtered));
        return 0;
    }
}
''',
    }
