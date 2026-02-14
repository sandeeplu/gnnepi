from jarvis.db.jsonutils import loadjson
def count_entries(file_name, single_value=False):
    data = loadjson(file_name)
    total = 0
    for entry in data:
        t_out = entry['target_out']
        if single_value:
            total += 1
        else:
            total += len(t_out)
    return total

train_total = count_entries("Train_results.json")
val_total = count_entries("Val_results.json")
test_total = count_entries("Test_results.json", single_value=True)
print("Entries count check:")
print(f"Train: {train_total}")
print(f"Validation: {val_total}")
print(f"Test: {test_total}")
print(f"Combined total: {train_total + val_total + test_total}")
