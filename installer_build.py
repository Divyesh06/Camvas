import subprocess


# Path to ISCC.exe
iscc_path = r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

# Path to your .iss script
iss_file = r"setup_generator.iss"

# Run the command
result = subprocess.run([iscc_path, iss_file], capture_output=True, text=True)

# Output result
print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)