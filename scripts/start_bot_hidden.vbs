' Start Trading Bot Hidden (No Terminal Window)
' Double-click this file to start the bot in background

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
WshShell.CurrentDirectory = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
WshShell.Run "pythonw dashboard.py", 0, False

' Show notification
MsgBox "Trading Bot Started!" & vbCrLf & vbCrLf & "Dashboard: http://localhost:5000" & vbCrLf & "The bot is running in background.", vbInformation, "AI Trading Bot"
