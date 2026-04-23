; Defaults — overridden at build time via /D flags from run_inno.py, which
; reads the live values out of pyproject.toml. Keeping them here means the
; .iss still works if ISCC is invoked directly.
#ifndef MyAppName
  #define MyAppName "Camvas"
#endif
#ifndef MyAppVersion
  #define MyAppVersion "1.0"
#endif
#ifndef MyAppPublisher
  #define MyAppPublisher "Camvas"
#endif
#ifndef MyBuildDir
  #define MyBuildDir "exe.win-amd64-3.12"
#endif

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultGroupName={#MyAppName}
DefaultDirName={pf}\{#MyAppName}
SourceDir=..
OutputDir=build\installer
OutputBaseFilename={#MyAppName}-{#MyAppVersion}
Compression=lzma
SolidCompression=yes
SetupIconFile=assets\Camvas.ico
AlwaysRestart=yes
CloseApplications=force


[Files]
Source: "build\{#MyBuildDir}\*"; DestDir: "{app}"; Flags: recursesubdirs

[Run]
Filename: "regsvr32.exe"; Parameters: "/s ""{app}\lib\softcam_python\softcam.dll"""; StatusMsg: "Registering dll..."; Flags: runhidden
Filename: "{app}\{#MyAppName}.exe"; StatusMsg: "Launching {#MyAppName}..."; Flags: postinstall

[UninstallRun]
Filename: "taskkill.exe"; Parameters: "/F /IM {#MyAppName}.exe /T"; Flags: runhidden; RunOnceId: "Kill{#MyAppName}"
Filename: "regsvr32.exe"; Parameters: "/u /s ""{app}\lib\softcam_python\softcam.dll"""; StatusMsg: "Unregistering dll..."; Flags: runhidden

[Tasks]
Name: "runonstartup"; Description: "Run {#MyAppName} on startup"; GroupDescription: "Additional options"; Flags: checkedonce
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional options"; Flags: unchecked

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; \
    ValueName: "DisplayName"; ValueType: string; ValueData: "{#MyAppName}"

Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; \
    ValueName: "UninstallString"; ValueType: string; ValueData: "{uninstallexe}"

Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; \
    ValueName: "DisplayIcon"; ValueType: string; ValueData: "{app}\{#MyAppName}.exe"

Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; \
    ValueName: "DisplayVersion"; ValueType: string; ValueData: "{#MyAppVersion}"

Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; \
    ValueName: "Publisher"; ValueType: string; ValueData: "{#MyAppPublisher}"

Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
    ValueName: "{#MyAppName}"; ValueType: string; ValueData: """{app}\{#MyAppName}.exe"" --startup"; Tasks: runonstartup

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppName}.exe"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"; IconFilename: "{app}\{#MyAppName}.exe"
Name: "{userdesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppName}.exe"; Tasks: desktopicon
