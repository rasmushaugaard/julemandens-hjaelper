import utils

annotations = utils.load_annotations()
ok = [a[1] for a in annotations]
print(sum(ok) / len(ok))