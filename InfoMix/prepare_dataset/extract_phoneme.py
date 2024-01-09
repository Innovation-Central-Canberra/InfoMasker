import os
import csv
from textgrids import TextGrid
from pydub import AudioSegment
from tqdm import tqdm
import re
from utils.mapping_dict import expand_contractions

from utils.parameters import *

# The original format of LibriSpeech is flac. Here we convert it into wav for convenience
def convert_FLAC_to_WAV(librispeech_path=LIBRISPEECH_PATH):
    for (dirpath, dirnames, filenames) in tqdm(os.walk(librispeech_path)):
        for filename in filenames:
            if filename.endswith(".flac"):
                filepath = os.path.join(dirpath, filename)
                (path, extension) = os.path.splitext(filepath)
                outputfilename = path + ".wav"
                audioin = AudioSegment.from_file(filepath, "flac")
                audioin.export(outputfilename, format="wav")
                os.system('rm -f '+filepath)

# 1. Make dir for each audio file in LibriSpeech and move the audio into the dir
# 2. Change audio filename into file.wav
# 3. Create transcript file for each audio and name as file.lab
def prepare_for_phoneme_extraction():
    # We only consider train-clean-100, train-clean-360, test-clean, dev-clean
    subsets = ['train-clean-100', 'train-clean-360', 'test-clean', 'dev-clean']
    for subset in subsets:
        # Set the root path for the dataset
        rootPath = os.path.join(LIBRISPEECH_PATH, subset)
        rootDirs = os.listdir(rootPath)
        numOfPeople = len(rootDirs)

        targetPath = os.path.join(PHONEME_INDEX_PATH, subset)

        # people_id - book_id - transcript_id
        transcriptPattern = '^[0-9]+-[0-9]+-[0-9]+'

        for people in tqdm(rootDirs):
            
            peoplePath = os.path.join(rootPath, people)
            peopleDirs = os.listdir(peoplePath)

            # make people dir in the target folder
            targetPeoplePath = os.path.join(targetPath, people)
            os.system('mkdir ' + targetPeoplePath)

            for book in peopleDirs:
                bookPath = os.path.join(peoplePath, book)

                # make book dir in the target folder
                targetBookpath = os.path.join(targetPeoplePath, book)
                os.system('mkdir ' + targetBookpath)

                # get the transcript file name
                transcriptFileName = people + '-' + book + '.trans.txt'
                transcriptFilePath = os.path.join(bookPath, transcriptFileName)

                transcriptFile = open(transcriptFilePath, "r")
                line = transcriptFile.readline()
                while line:
                    matchResult = re.match(transcriptPattern, line)
                    matchResult = matchResult.group()
                    
                    # make dir for a single transcript
                    singleTranscriptPath = os.path.join(targetBookpath, matchResult)
                    os.system('mkdir ' + singleTranscriptPath)
                    
                    transcript = line[len(matchResult) + 1 :-1]
                    refined_transcript = expand_contractions(transcript.lower()).replace('\'s', '')
                    refined_transcript = refined_transcript.replace('s\'', '').upper()

                    lab_file = open(singleTranscriptPath+'/file.lab','w')
                    lab_file.write(refined_transcript)
                    lab_file.close()

                    os.system('cp ' + bookPath + '/' +matchResult + '.wav ' + singleTranscriptPath + '/file.wav')

                    line = transcriptFile.readline()

                transcriptFile.close()

# 1. The aligner will first generate a TextGrid file for each audio file, containing vowels and consonants
# 2. Get the start time stamp and end time stamp for each phoneme and store into vowel.csv and consonant.csv for each audio file.
def extract_phoneme_for_dataset():
    # We only consider train-clean-100, train-clean-360, test-clean, dev-clean
    subsets = ['train-clean-100', 'train-clean-360', 'test-clean', 'dev-clean']
    for subset in subsets:
        root_path = os.path.join(PHONEME_INDEX_PATH, subset)
        dirs = os.listdir(root_path)
        for item in dirs:
            people_path = os.path.join(root_path, item)
            people_folders = os.listdir(people_path)
            for book in people_folders:
                book_path = os.path.join(people_path, book)
                book_folder = os.listdir(book_path)
                for transcript in book_folder:
                    folder_path = os.path.join(book_path, transcript)
                    os.chdir(ALIGNER_PATH)
                    extract_phoneme(folder_path)

def extract_phoneme(folder_path):

    # Try to align the wav file with transcript.
    # OOV ERROR means there are words in the file but not in the dict
    # Todo: What to do if there are words in the file but not in the dict
    try:
        os.system('python -m aligner -r eng.zip -a ' + folder_path + ' -d eng.dict')
    except:
        print('OOV ERROR!')
    
    # Read the TextGrid file and generate csv file for vowels and consonants
    ## We only consider the following vowels and consonants
    vowels = ["AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2"," AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "OW0", "OW1", "OW2","OY0", "OY1", "OY2", "UH0", "UH1", "UH2", "UW0", "UW1", "UW2"] 
    consonants = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "S    H", "T", "TH", "V", "W", "Y", "Z", "ZH"]

    ## Read the TextGrid file
    try:
        grid = TextGrid(folder_path + '/file.TextGrid')
    except:
        os.system('rm -rf '+ folder_path)
        print('NO FILE ERROR!')
    else:
        phoneme = grid['phones']
        
        # write the vowels and consontants to two csv files 
        vowelFile = open(folder_path + '/vowel.csv', 'w', newline = '')
        vowelWriter = csv.writer(vowelFile)
        
        consonantFile = open(folder_path + '/consonant.csv', 'w', newline='')
        consonantWriter = csv.writer(consonantFile)
        
        for item in phoneme:
            if(item.text in vowels):
                vowelWriter.writerow([item.text] + [str(item.xmin)] + [str(item.xmax)])
            elif(item.text in consonants):
                consonantWriter.writerow([item.text] + [str(item.xmin)] + [str(item.xmax)])

if __name__ == '__main__':
    convert_FLAC_to_WAV()
    prepare_for_phoneme_extraction()
    extract_phoneme_for_dataset()
