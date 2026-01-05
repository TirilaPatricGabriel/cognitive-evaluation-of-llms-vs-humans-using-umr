from zuco_utils import *

if __name__ == "__main__":
    zuco_eye_file = "zuco2_eye.csv"
    zuco_sentences_file = "zuco2_sentences.csv"

    create_zuco_sentences_file(zuco_eye_file, zuco_sentences_file)
    
    create_average_subject_file(zuco_eye_file, "zuco2_average_subject.csv")
    
    create_average_sentence_level_file("zuco2_average_subject.csv", "zuco2_average_sentence_level.csv")
    
    create_participants_sentence_level_file(zuco_eye_file, "zuco2_participants_sentence_level.csv")
    