import os


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))

data_directory = r"D:\Downloads\ASL"

TRAIN_DIR=os.path.join(data_directory, r"asl_alphabet_train\asl_alphabet_train")
TEST_DIR=os.path.join(data_directory, r"asl_alphabet_test\asl_alphabet_test")
data_dir = os.path.join(get_project_root(), "data")
checkpoints_dir = os.path.join(get_project_root(), "checkpoints")

index_to_label = {i: chr(ord('A')+i) for i in range(26)}
index_to_label.update({26:"del", 27:"nothing", 28:"space"})