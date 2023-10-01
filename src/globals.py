DATA_PATH = "data/"


GRADES = []
GRADE_DICT = {}
INV_GRADE_DICT = {}
grade_ct = 0
for num in range(6, 9):
    for let in ["A", "B", "C"]:
        for pls in ["", "+"]:
            GRADES.append(str(num) + let + pls)
            GRADE_DICT[str(num) + let + pls] = grade_ct
            INV_GRADE_DICT[grade_ct] = str(num) + let + pls
            grade_ct += 1
