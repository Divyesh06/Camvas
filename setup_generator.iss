[Setup]
AppName=Camvas
AppVersion=1.0
DefaultGroupName=Camvas
AppPublisher=Camvas
AppPublisherURL=https://camvas.geeke.app
AppSupportURL=https://camvas.geeke.app/support
DefaultDirName={pf}\Camvas
OutputDir=installer
OutputBaseFilename=Camvas
Compression=lzma
SolidCompression=yes


[Files]
Source: "build\exe.win-amd64-3.12\*"; DestDir: "{app}"; Flags: recursesubdirs

[Run]
Filename: "regsvr32.exe"; Parameters: "/s ""{app}\lib\softcam_python\softcam.dll"""; StatusMsg: "Registering dll..."; Flags: runhidden
Filename: "{app}\Camvas.exe"; StatusMsg: "Launching Camvas..."; Flags: postinstall

[Tasks]
Name: "runonstartup"; Description: "Run Camvas on startup"; GroupDescription: "Additional options"; Flags: checkedonce
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional options"; Flags: unchecked

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\Camvas"; \
    ValueName: "DisplayName"; ValueType: string; ValueData: "Camvas"

Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\Camvas"; \
    ValueName: "UninstallString"; ValueType: string; ValueData: "{uninstallexe}"

Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\Camvas"; \
    ValueName: "DisplayIcon"; ValueType: string; ValueData: "{app}\app.exe"

Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\Camvas"; \
    ValueName: "DisplayVersion"; ValueType: string; ValueData: "1.0"

Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\Camvas"; \
    ValueName: "Publisher"; ValueType: string; ValueData: "Camvas"
    
[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
    ValueName: "Camvas"; ValueType: string; ValueData: """{app}\Camvas.exe"" --startup"; Tasks: runonstartup

[Icons]
Name: "{group}\Camvas"; Filename: "{app}\Camvas.exe"
Name: "{group}\Uninstall Camvas"; Filename: "{uninstallexe}"; IconFilename: "{app}\Camvas.exe"
Name: "{userdesktop}\Camvas"; Filename: "{app}\Camvas.exe"; Tasks: desktopicon