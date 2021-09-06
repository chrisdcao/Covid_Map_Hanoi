import fileinput

with open("db_add_info.txt", "r") as f:
    info_lines = f.read()

print(info_lines)

with fileinput.input("add_db.py", inplace = True) as f_write:
    for line in f_write:
        if "VALUES" in line:
            line = line.replace(line, line + info_lines)
        print(line, end='')

