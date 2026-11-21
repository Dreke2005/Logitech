using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Loupedeck.PaperDeskPlugin;

public abstract class PaperAppFolder : PluginDynamicFolder
{
    private static readonly Dictionary<string, string[][]> Menus = LoadMenus();
    private readonly string app;
    private string? displayedMenu;

    protected PaperAppFolder(string app, string name)
    {
        this.app = app;
        DisplayName = name;
        GroupName = "Paper apps";
    }

    private string ActiveMenu => app == "vscode" && PaperDeskPlugin.Displayed?.menu == "vscode_languages"
        ? "vscode_languages"
        : app;

    public override bool Activate()
    {
        displayedMenu = ActiveMenu;
        PaperDeskPlugin.KeyLabelsChanged += RedrawMenu;
        Send("home", app);
        return true;
    }

    public override bool Deactivate()
    {
        PaperDeskPlugin.KeyLabelsChanged -= RedrawMenu;
        displayedMenu = null;
        return true;
    }

    private void RedrawMenu()
    {
        string menu = ActiveMenu;
        if (menu != displayedMenu)
        {
            displayedMenu = menu;
            ButtonActionNamesChanged();
            return;
        }
        if (Menus[menu].Any(button => button[0] == "camera"))
        {
            CommandImageChanged($"{menu}/camera");
        }
    }

    public override PluginDynamicFolderNavigation GetNavigationArea(DeviceType device) =>
        PluginDynamicFolderNavigation.None;

    public override IEnumerable<string> GetButtonPressActionNames(DeviceType device)
    {
        string menu = ActiveMenu;
        return Menus[menu].Select(button => CreateCommandName($"{menu}/{button[0]}"));
    }

    public override string GetCommandDisplayName(string parameter, PluginImageSize imageSize) => string.Empty;

    public override string GetButtonDisplayName(PluginImageSize imageSize) => string.Empty;

    public override BitmapImage GetButtonImage(PluginImageSize imageSize) => PaperIcons.Read(app, imageSize);

    public override BitmapImage GetCommandImage(string parameter, PluginImageSize imageSize)
    {
        var parts = parameter.Split('/', 2);
        string[]? button = parts.Length == 2 && Menus.TryGetValue(parts[0], out var buttons)
            ? buttons.FirstOrDefault(item => item[0] == parts[1])
            : null;
        return PaperIcons.Read(PaperIcons.ForButton(button), imageSize);
    }

    public override void RunCommand(string parameter)
    {
        var parts = parameter.Split('/', 2);
        if (parts.Length == 2 && (parts[0] == app || (app == "vscode" && parts[0] == "vscode_languages"))
            && Menus[parts[0]].Any(button => button[0] == parts[1]))
        {
            Send(parts[0], parts[1]);
            if (parts[1] == "home")
            {
                Close();
            }
        }
    }

    private void Send(string menu, string action)
    {
        try
        {
            this.Plugin.Log.Info($"Paper folder input {menu}/{action}.");
            var state = PaperDeskPlugin.Displayed;
            if (state is null)
            {
                PaperKey.OpenWorkbench(app);
                return;
            }
            PaperDeskPlugin.WriteCommand(new { state.session, menu, action });
        }
        catch (IOException error)
        {
            this.Plugin.Log.Error(error.Message);
        }
        catch (UnauthorizedAccessException error)
        {
            this.Plugin.Log.Error(error.Message);
        }
        catch (System.ComponentModel.Win32Exception error)
        {
            this.Plugin.Log.Error(error.Message);
        }
    }

    private static Dictionary<string, string[][]> LoadMenus()
    {
        using var stream = typeof(PaperAppFolder).Assembly.GetManifestResourceStream("PaperMenus.json")!;
        return JsonSerializer.Deserialize<Dictionary<string, string[][]>>(stream)!;
    }

    internal static string IconForSlot(string? menu, int slot)
    {
        return Menus.TryGetValue(menu ?? "home", out var buttons) && slot < buttons.Length
            ? PaperIcons.ForButton(buttons[slot])
            : "blank";
    }
}

public sealed class VsCodePaperFolder : PaperAppFolder
{
    public VsCodePaperFolder() : base("vscode", "VS Code") { }
}

public sealed class MatlabPaperFolder : PaperAppFolder
{
    public MatlabPaperFolder() : base("matlab", "MATLAB") { }
}

public sealed class LatexPaperFolder : PaperAppFolder
{
    public LatexPaperFolder() : base("latex", "LaTeX") { }
}

public sealed class ChatGptPaperFolder : PaperAppFolder
{
    public ChatGptPaperFolder() : base("chatgpt", "ChatGPT") { }
}

public sealed class KiCadPaperFolder : PaperAppFolder
{
    public KiCadPaperFolder() : base("kicad", "KiCad") { }
}

public sealed class GitPaperFolder : PaperAppFolder
{
    public GitPaperFolder() : base("git", "Git") { }
}
