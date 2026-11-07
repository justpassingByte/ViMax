; Inno Setup Script for ViMax Studio
[Setup]
AppName=ViMax Studio
AppVersion=1.2.0
DefaultDirName={userappdata}\ViMax Studio
DefaultGroupName=ViMax Studio
OutputBaseFilename=ViMax_Studio_Setup
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
DisableProgramGroupPage=yes
PrivilegesRequired=lowest

[Tasks]
Name: "desktopicon"; Description: "Tạo biểu tượng ngoài màn hình Desktop"; GroupDescription: "Tùy chọn bổ sung:"

[Files]
Source: "ViMax_Studio.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "run_web.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "HUONG_DAN_SU_DUNG.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "bin\*"; DestDir: "{app}\bin"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: ".venv\*"; DestDir: "{app}\.venv"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "web\*"; DestDir: "{app}\web"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "agents\*"; DestDir: "{app}\agents"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "agent_runtime\*"; DestDir: "{app}\agent_runtime"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "configs\*"; DestDir: "{app}\configs"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "interfaces\*"; DestDir: "{app}\interfaces"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "pipelines\*"; DestDir: "{app}\pipelines"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "prompts\*"; DestDir: "{app}\prompts"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "tools\*"; DestDir: "{app}\tools"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "utils\*"; DestDir: "{app}\utils"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "assets\*"; DestDir: "{app}\assets"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\ViMax Studio"; Filename: "{app}\ViMax_Studio.exe"
Name: "{autodesktop}\ViMax Studio"; Filename: "{app}\ViMax_Studio.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ViMax_Studio.exe"; Description: "Khởi động ViMax Studio ngay"; Flags: nowait postinstall skipifsilent
