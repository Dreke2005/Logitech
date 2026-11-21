using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text.Json;
using System.Threading;

namespace Loupedeck.PaperDeskPlugin;

public sealed class PaperDeskPlugin : Plugin
{
    public override bool UsesApplicationApiOnly => true;
    public override bool HasNoApplication => true;

    internal static readonly string DataPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "LogiInsightPaper");
    internal static event Action? KeyLabelsChanged;
    internal static volatile KeypadState? Displayed;
    private Timer? timer;

    public override void Load()
    {
        timer = new Timer(_ => Refresh(), null, TimeSpan.FromMilliseconds(350), TimeSpan.FromMilliseconds(350));
    }

    public override void Unload()
    {
        timer?.Dispose();
        Displayed = null;
    }

    private void Refresh()
    {
        var current = Read<KeypadState>("state.json");
        if (current?.labels?.Length != 9 ||
            Math.Abs(DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - current.updated) > 6000)
        {
            current = null;
        }
        var previous = Displayed;
        Displayed = current;
        if (previous?.session != current?.session || previous?.revision != current?.revision
            || previous?.cameraActive != current?.cameraActive)
        {
            this.Log.Info(current is null ? "Paper workbench closed." : $"Paper menu revision {current.revision}.");
            KeyLabelsChanged?.Invoke();
        }
    }

    internal static T? Read<T>(string file)
    {
        try
        {
            using var stream = new FileStream(Path.Combine(DataPath, file), FileMode.Open, FileAccess.Read,
                FileShare.ReadWrite | FileShare.Delete);
            return JsonSerializer.Deserialize<T>(stream);
        }
        catch (IOException) { return default; }
        catch (UnauthorizedAccessException) { return default; }
        catch (JsonException) { return default; }
    }

    internal static void WriteCommand(object command)
    {
        var folder = Path.Combine(DataPath, "commands");
        Directory.CreateDirectory(folder);
        var file = Path.Combine(folder, $"{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds():D13}-{Guid.NewGuid():N}");
        File.WriteAllText(file + ".tmp", JsonSerializer.Serialize(command));
        File.Move(file + ".tmp", file + ".json");
    }

    internal sealed record KeypadState(string session, int revision, long updated, string[] labels,
        string? menu, bool cameraActive = false);
    internal sealed record Launcher(string python, string script);
}

public sealed class PaperDeskApplication : ClientApplication
{
    protected override string GetProcessName() => "";
    protected override string GetBundleName() => "";
    public override ClientApplicationStatus GetApplicationStatus() => ClientApplicationStatus.Unknown;
}

public sealed class PaperKey : PluginDynamicCommand
{
    private static long lastLaunch = -4000;

    public PaperKey() : base()
    {
        for (int slot = 0; slot < 9; ++slot)
        {
            AddParameter(slot.ToString(CultureInfo.InvariantCulture), $"Paper key {slot + 1}", "Paper desk");
        }
    }

    protected override bool OnLoad()
    {
        PaperDeskPlugin.KeyLabelsChanged += Redraw;
        return true;
    }

    protected override bool OnUnload()
    {
        PaperDeskPlugin.KeyLabelsChanged -= Redraw;
        return true;
    }

    private void Redraw()
    {
        for (int slot = 0; slot < 9; ++slot)
        {
            ActionImageChanged(slot.ToString(CultureInfo.InvariantCulture));
        }
    }

    private static bool TrySlot(string parameter, out int slot) =>
        int.TryParse(parameter, NumberStyles.None, CultureInfo.InvariantCulture, out slot) && slot is >= 0 and < 9;

    protected override string GetCommandDisplayName(string actionParameter, PluginImageSize imageSize) =>
        string.Empty;

    protected override BitmapImage GetCommandImage(string actionParameter, PluginImageSize imageSize)
    {
        string icon = TrySlot(actionParameter, out var slot)
            ? PaperDeskPlugin.Displayed is { } state
                ? PaperAppFolder.IconForSlot(state.menu, slot)
                : slot == 0 ? "open" : "blank"
            : "blank";
        return PaperIcons.Read(icon, imageSize);
    }

    protected override void RunCommand(string actionParameter)
    {
        if (!TrySlot(actionParameter, out int slot))
        {
            return;
        }
        var state = PaperDeskPlugin.Displayed;
        try
        {
            if (state is null)
            {
                if (slot == 0)
                {
                    OpenWorkbench();
                }
                return;
            }
            if (state.labels[slot] == "")
            {
                return;
            }
            PaperDeskPlugin.WriteCommand(new { state.session, state.revision, slot });
        }
        catch (IOException error) { this.Plugin.Log.Error(error.Message); }
        catch (UnauthorizedAccessException error) { this.Plugin.Log.Error(error.Message); }
        catch (System.ComponentModel.Win32Exception error) { this.Plugin.Log.Error(error.Message); }
    }

    internal static void OpenWorkbench(string? route = null)
    {
        long now = Environment.TickCount64;
        long previous = Interlocked.Read(ref lastLaunch);
        if (now - previous < 4000 || Interlocked.CompareExchange(ref lastLaunch, now, previous) != previous)
        {
            return;
        }
        var launcher = PaperDeskPlugin.Read<PaperDeskPlugin.Launcher>("launcher.json");
        if (launcher is null || !File.Exists(launcher.python) || !File.Exists(launcher.script))
        {
            return;
        }
        var start = new ProcessStartInfo(launcher.python) {
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(launcher.script)!,
        };
        start.ArgumentList.Add(launcher.script);
        if (route is not null)
        {
            start.ArgumentList.Add("--route");
            start.ArgumentList.Add(route);
        }
        using var process = Process.Start(start);
    }
}
