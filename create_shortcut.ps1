# Creates the Batscribe shortcut (app folder + Start menu) with its icon and AppUserModelID.
# The ID must match APP_ID in batscribe_gui.py: it tells Windows that the running window *is*
# this shortcut, so pinning it to the taskbar pins the shortcut (pythonw + script), not "Python".
#
# Run once (and again if the folder moves):  powershell -ExecutionPolicy Bypass -File create_shortcut.ps1
$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$appId = 'Batscribe.Captions'

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;

[ComImport, Guid("00021401-0000-0000-C000-000000000046")] class CShellLink {}

[ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("000214F9-0000-0000-C000-000000000046")]
interface IShellLinkW {
    void GetPath([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder f, int c, IntPtr fd, uint fl);
    void GetIDList(out IntPtr p);
    void SetIDList(IntPtr p);
    void GetDescription([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder s, int c);
    void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string s);
    void GetWorkingDirectory([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder s, int c);
    void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string s);
    void GetArguments([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder s, int c);
    void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string s);
    void GetHotkey(out short h);
    void SetHotkey(short h);
    void GetShowCmd(out int i);
    void SetShowCmd(int i);
    void GetIconLocation([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder s, int c, out int i);
    void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string s, int i);
    void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string s, uint r);
    void Resolve(IntPtr h, uint f);
    void SetPath([MarshalAs(UnmanagedType.LPWStr)] string s);
}

[StructLayout(LayoutKind.Sequential, Pack = 4)] struct PropertyKey { public Guid fmtid; public uint pid; }
[StructLayout(LayoutKind.Explicit)] struct PropVariant { [FieldOffset(0)] public ushort vt; [FieldOffset(8)] public IntPtr p; }

[ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")]
interface IPropertyStore {
    void GetCount(out uint c);
    void GetAt(uint i, out PropertyKey k);
    void GetValue(ref PropertyKey k, out PropVariant v);
    void SetValue(ref PropertyKey k, ref PropVariant v);
    void Commit();
}

[ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("0000010b-0000-0000-C000-000000000046")]
interface IPersistFile {
    void GetClassID(out Guid g);
    [PreserveSig] int IsDirty();
    void Load([MarshalAs(UnmanagedType.LPWStr)] string f, uint m);
    void Save([MarshalAs(UnmanagedType.LPWStr)] string f, bool remember);
    void SaveCompleted([MarshalAs(UnmanagedType.LPWStr)] string f);
    void GetCurFile([MarshalAs(UnmanagedType.LPWStr)] out string f);
}

public static class BatscribeLnk {
    public static void Create(string lnk, string target, string args, string dir, string icon, string appId) {
        var link = (IShellLinkW)new CShellLink();
        link.SetPath(target);
        link.SetArguments(args);
        link.SetWorkingDirectory(dir);
        link.SetIconLocation(icon, 0);
        link.SetDescription("Batscribe - caption videos to .srt");
        var key = new PropertyKey { fmtid = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), pid = 5 };  // System.AppUserModel.ID
        var value = new PropVariant { vt = 31, p = Marshal.StringToCoTaskMemUni(appId) };                 // VT_LPWSTR
        try {
            var store = (IPropertyStore)link;
            store.SetValue(ref key, ref value);
            store.Commit();
            ((IPersistFile)link).Save(lnk, true);
        } finally {
            Marshal.FreeCoTaskMem(value.p);
        }
    }
}
'@

$target = Join-Path $repo '.venv\Scripts\pythonw.exe'
if (-not (Test-Path $target)) { throw "Missing $target - create the venv first (see requirements.txt)." }
$arguments = '"' + (Join-Path $repo 'batscribe_gui.py') + '"'
$icon = Join-Path $repo 'batscribe.ico'
$startMenu = Join-Path ([Environment]::GetFolderPath('Programs')) 'Batscribe.lnk'

foreach ($lnk in (Join-Path $repo 'Batscribe.lnk'), $startMenu) {
    [BatscribeLnk]::Create($lnk, $target, $arguments, $repo, $icon, $appId)
    "Created $lnk"
}
