import sys

arg0 = sys.argv[0]
arg1 = sys.argv[1]
session_ids = sys.argv[2]

session_ids = session_ids.split(",")
print(session_ids)
session_ids[0] = session_ids[0].replace("[", "")
print(session_ids)
session_ids[-1] = session_ids[-1].replace("]", "")
print(session_ids)
