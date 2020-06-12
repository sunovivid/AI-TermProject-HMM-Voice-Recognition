# hmmlearn 패키지 사용해서 결과 확인하고 직접 구현으로 넘어갈까?
import pprint
import operator
from functools import reduce
from collections import OrderedDict
import os
import sys
import math
import pickle
import time

import multiprocessing as mp
from functools import partial

import numpy as np
# np.set_printoptions(formatter={"float_kind": lambda x: "{0:+0.2f}".format(x)})# 4.2

#TODO 1. 음소 HMM(hmm.txt/hmn.h), 발음사전(dictionary.txt) 이용해 단어 HMM 구축
#TODO 2. language model(bigram.txt) 추가해 임의의 단어 시퀀스 HMM 구축

#HYPERPARAMETER
LANGUAGE_MODEL_WEIGHT = 2

#CONSTANT
N_STATE	= 3
N_PDF = 2
PI = 3.14
N_DIMENSION = 39

FILEPATH_UNIGRAM = './data/unigram.txt'
FILEPATH_BIGRAM = './data/bigram.txt'
FILEPATH_DICTIONARY = './data/dictionary.txt'
FILEPATH_HMM = './data/hmm.txt'
FILEPATH_DATA = './data'
FILEPATH_TEST = './tst'
FILEPATH_RECOGNIZED = './'
PHONES = list(
    {"ey", "t", "f", "ay", "v", "ao", "n", "ow", "w", "ah", "s", "eh", "ih", "k", "th", "r", "iy", "uw", "z", "sp", "sil"})
NUM_OF_PHONES = len(PHONES)

# class pdf(): #multivariate gaussian model
#     def __init__(self, weight, mean, var):
#         self.weight = weight
#         self.mean = mean
#         self.var = var

def np_pprint(arr):
    print("type:{}".format(type(arr)))
    print("shape: {}, dimension: {}, dtype:{}".format(arr.shape, arr.ndim, arr.dtype))
    print("Array's Data:\n", arr)

def print_tp_with_state(hmm, tp):
    print("    ", end='')
    prev_word = ""
    for i, state_dict in enumerate(hmm.states):
        if prev_word != state_dict['word']:
            print("|" + "%3s" % state_dict['phone'], end='')
            prev_word = state_dict['word']
        else:
            print("%4s" % state_dict['phone'], end='')
    print()
    prev_word = ""
    for t, state_dict in enumerate(hmm.states):
        if prev_word != state_dict['word']:
            print("{0:->4}".format(state_dict['phone']), end='')
            prev_word = state_dict['word']
        else:
            print("%4s" % state_dict['phone'], end='')
        print(np.array2string(tp[t], max_line_width=np.inf))

def print_with_state(hmm, arr):
    assert (hmm.NUM_OF_STATES == len(arr))
    print("type:{}".format(type(arr)))
    print("shape: {}, dimension: {}, dtype:{}".format(arr.shape, arr.ndim, arr.dtype))

    prev_word = ""
    for i, z in enumerate(zip(hmm.states,arr)):
        state_dict, arr_elem = z
        if prev_word != state_dict['word']:
            print(f"\n[{state_dict['word']}] ", end='')
            prev_word = state_dict['word']
        if arr.dtype == 'f8':
            print(f"{'{:.2f}'.format(arr_elem)}({state_dict['phone']},{i}) ", end='')
        else:
            print(f"{arr_elem}({state_dict['phone']},{i}) ", end='')
    print()

class StatePdf(): #Gaussian mixture model emission probability for certain state
    def __init__(self, w1, w2, mean1, mean2, var1, var2):
        self.mean1 = mean1
        self.mean2 = mean2
        self.var1 = var1
        self.var2 = var2
        self.log_a = math.log(w1) - (N_DIMENSION / 2) * math.log(2 * PI) - math.log(reduce(operator.mul, var1)) / 2 #속도 향상을 위해 미리 계산
        self.log_b = math.log(w1) - (N_DIMENSION / 2) * math.log(2 * PI) - math.log(reduce(operator.mul, var2)) / 2

class Hmm():
    def __init__(self, prior_prob, tp, states):
        self.logscale_prior_prob = np.log(prior_prob)
        self.logscale_tp = np.log(tp)
        self.states = states

    @property
    def NUM_OF_STATES(self):
        return len(self.states) #state idx = 0 to n(116)이나, states에 시작 state는 포함되어 있지 않으므로 +1

    def log_sensor_model(self, evidence, state):
        state_pdf = self.states[state]["statePdf"]
        if state_pdf is None: # for state index 0
            return 0
        log_a = state_pdf.log_a # state.a = w1 / COEF / reduce(operator.mul, sigma1) 미리 계산해둔 값 불러옴
        log_b = state_pdf.log_b # state.b = w2 / COEF / reduce(operator.mul, sigma2)
        p = -0.5 * sum( map ( operator.truediv, map ( lambda x: pow(x,2), map ( operator.sub, evidence, state_pdf.mean1)), state_pdf.var1) )
        q = -0.5 * sum( map ( operator.truediv, map ( lambda x: pow(x,2), map ( operator.sub, evidence, state_pdf.mean2)), state_pdf.var2) )
        return log_a + p + math.log(1 + (math.exp(log_a - log_b + q - p)))

    def log_sensor_arr(self, evidence):
        arr = np.zeros(self.NUM_OF_STATES, np.float)
        for state in range(self.NUM_OF_STATES):
            arr[state] = self.log_sensor_model(evidence, state) #state 1부터 n까지의 pdf를 arr 0부터 n-1에 저장
        return arr

    def print_state(self):
        prev_word = ""
        print("STATES")
        for i, state_dict in enumerate(self.states):
            if prev_word != state_dict['word']:
                print()
                print(f"[{state_dict['word']}]",end='')
                prev_word = state_dict['word']
            print(f" {i}({state_dict['phone']})", end=' /')

    def print_tp(self):
        print()
        print("TP")
        print_tp_with_state(self, self.logscale_tp)

def get_data():
    # get unigram
    unigram = OrderedDict()  # unigram: list of {string: float}
    with open(FILEPATH_UNIGRAM, 'r') as bigram_file:
        for line in bigram_file:
            str, prob = line.split()
            unigram[str] = float(prob)

    # get bigram
    bigram = OrderedDict() #bigram: list of {(string, string): float}
    with open(FILEPATH_BIGRAM, 'r') as bigram_file:
        for line in bigram_file:
            str1, str2, prob = line.split()
            bigram[(str1,str2)] = float(prob)

    # get dictionary
    dictionary = OrderedDict() #dictionary: list of {"eight": ["ey", "t", "sp"]}
    with open(FILEPATH_DICTIONARY, 'r') as dictionary_file:
        for line in dictionary_file:
            dictionary[line.split()[0]] = line.split()[1:] #zero는 마지막 걸로 덮어씌워지고 나머지 수동으로 입력

    # get phone hmm
    phone_hmm_data = OrderedDict() #phone_hmm_data: {"f": {state_pdf_list: [StatePdf객체1, StatePdf객체2, ...], tp: 행렬}, ...}
    with open(FILEPATH_HMM, 'r') as hmm_file:
        for _ in range(NUM_OF_PHONES):
            phone_name = hmm_file.readline().split('"')[1]
            assert(hmm_file.readline().split()[0] == "<BEGINHMM>")
            num_of_state = int(hmm_file.readline().split()[1]) - 2 #<NUMSTATES>
            assert(num_of_state==3 or num_of_state==1)

            state_pdf_list = []

            for index in range(num_of_state):

                assert(int(hmm_file.readline().split()[1]) == index+2) #<STATE>
                num_of_mix = int(hmm_file.readline().split()[1]) #<NUMMIXES>
                assert(num_of_mix==2)

                # for _ in range(num_of_mix): num of mix가 일반화된 버전
                #
                #     weight = float(hmm_file.readline().split()[2]) #<MIXTURE>
                #     assert(int(hmm_file.readline().split()[1]) == N_DIMENSION) #<MEAN>
                #     mean = list(map(float,hmm_file.readline().split()))
                #     assert(int(hmm_file.readline().split()[1]) == N_DIMENSION) #<VARIANCE>
                #     var = list(map(float,hmm_file.readline().split()))

                #num of mix = 2 고정
                w1 = float(hmm_file.readline().split()[2]) #<MIXTURE>
                assert(int(hmm_file.readline().split()[1]) == N_DIMENSION) #<MEAN>
                mean1 = list(map(float,hmm_file.readline().split()))
                assert(int(hmm_file.readline().split()[1]) == N_DIMENSION) #<VARIANCE>
                var1 = list(map(float,hmm_file.readline().split()))

                w2 = float(hmm_file.readline().split()[2]) #<MIXTURE>
                assert(int(hmm_file.readline().split()[1]) == N_DIMENSION) #<MEAN>
                mean2 = list(map(float,hmm_file.readline().split()))
                assert(int(hmm_file.readline().split()[1]) == N_DIMENSION) #<VARIANCE>
                var2 = list(map(float,hmm_file.readline().split()))

                state_pdf_list.append(StatePdf(w1, w2, mean1, mean2, var1, var2))

            tp_n = int(hmm_file.readline().split()[1]) #<TRANSP>
            assert(tp_n == N_STATE+2 or tp_n == 1+2)  #1(+2) for optional silence
            tp = []
            for _ in range(tp_n):
                tp_line = list(map(float,hmm_file.readline().split()))
                tp.append(tp_line)
            assert(hmm_file.readline().split()[0] == "<ENDHMM>")

            phone_hmm_data[phone_name] = {"state_pdf_list" : state_pdf_list, "tp": tp}
    assert(NUM_OF_PHONES == len(phone_hmm_data))

    return unigram, bigram, dictionary, phone_hmm_data

def get_test_data():
    if 'test_data.p' in os.listdir(FILEPATH_DATA):
        print("Importing test data..")
        with open(os.path.join(FILEPATH_DATA,'test_data.p'), 'rb') as test_data_file:
            test_data = pickle.load(test_data_file)
            print("Imported.")
            return test_data

    test_data = OrderedDict() #{"tst/f/ak/1393387":numpy 객체}
    for folder_sex in os.listdir(FILEPATH_TEST):
        for folder_inner in os.listdir(f"{FILEPATH_TEST}/{folder_sex}"):
            for filename in os.listdir(f"{FILEPATH_TEST}/{folder_sex}/{folder_inner}"):
                with open(f"{FILEPATH_TEST}/{folder_sex}/{folder_inner}/{filename}", 'r') as MFCC_file:
                    num_of_vec, dimension = map(int, MFCC_file.readline().split())
                    print(f"Processing test/{folder_sex}/{folder_inner}/{filename}..")
                    assert (dimension == N_DIMENSION)
                    MFCC_vec_array = np.empty((num_of_vec, dimension), 'f8')
                    for i, line in enumerate(MFCC_file):
                        MFCC_vec_array[i] = list(map(float,line.split()))
                    test_data[f"{FILEPATH_TEST}/{folder_sex}/{folder_inner}/{filename}"] = MFCC_vec_array

    with open(os.path.join(FILEPATH_DATA,'test_data.p'),'wb') as test_data_file:
        pickle.dump(test_data, test_data_file)

    return test_data

def construct_word_seq_hmm(unigram, bigram, dictionary, phone_hmm_data):
    # Construct word seq hmm from data

    word_begin_idx_dict = OrderedDict() #{"<s>":1, "eight":4, "five":13, ...}
    word_end_idx_dict = OrderedDict()

    idx = 0
    states = []

    # pprint.pprint(dictionary)
    for word, phone_list in dictionary.items(): #단어 시작 인덱스 딕셔너리 생성
        word_begin_idx_dict[word] = idx
        idx += sum(map(lambda ph: len(phone_hmm_data[ph]["state_pdf_list"]), phone_list))
        word_end_idx_dict[word] = idx - 1
        # print(word, word_begin_idx_dict[word], word_end_idx_dict[word] )

    NUM_OF_STATES = idx + 2 + 1 #zero의 ih 경로 +3개 (idx가 새 ih 경로의 첫번째 state를 가리키고 있으므로 +2)
    last_sp_skip_prob = OrderedDict()
    last_sp_end_prob = OrderedDict()

    tp = np.zeros((NUM_OF_STATES, NUM_OF_STATES),float)
    is_zero_first = True

    prior_prob = np.zeros(NUM_OF_STATES)

    for word, phone_list in dictionary.items(): #word hmm 구성
        # print(word)
        if word == "zero":
            if not is_zero_first:
                continue
            is_zero_first = False

        idx = word_begin_idx_dict[word]

        prev_phone_idx = -1
        # prev_phone_exit_prob = unigram[word] #unigram
        prior_prob[idx] = unigram[word]

        for phone in phone_list:
            # print(f"\t{phone}")
            state_pdf_list = phone_hmm_data[phone]["state_pdf_list"]
            phone_tp = phone_hmm_data[phone]["tp"]

            for i, statePdf in enumerate(state_pdf_list): #states에 statePdf와 정보 객체 추가
                states.append({"word": word, "phone": phone, "idx": i, "statePdf":statePdf})

            if word == "zero":
                if phone == "z":
                    zero_z_end_idx = idx + 2
                elif phone == "r":
                    zero_r_begin_idx = idx

            if word != "<s>" and phone == "sp": #<s>가 아닌 단어의 끝
                tp[prev_phone_idx][idx] = prev_phone_exit_prob * phone_tp[0][1]
                tp[idx][idx] = phone_tp[1][1]
                last_sp_end_prob[word] = tp[1][2]
                last_sp_skip_prob[word] = tp[0][2]
            else:
                if prev_phone_idx != -1:
                    tp[prev_phone_idx][idx] = prev_phone_exit_prob
                tp[idx][idx] = phone_tp[1][1]
                tp[idx][idx+1] = phone_tp[1][2]
                tp[idx+1][idx+1] = phone_tp[2][2]
                tp[idx+1][idx+2] = phone_tp[2][3]
                tp[idx+2][idx+2] = phone_tp[3][3]
                if phone == "sil":
                    tp[idx][idx+2] = phone_tp[1][3]
                    tp[idx+2][idx] = phone_tp[3][1]
                    last_sp_end_prob[word] = tp[3][4]
                else: #check
                    prev_phone_idx = idx+2
                    prev_phone_exit_prob = phone_tp[3][4]

            idx += len(state_pdf_list)
            assert (len(states) == idx)

    # 예외 케이스: zero의 "ih" 경로 생성

    prob_z_branch = tp[zero_z_end_idx][zero_z_end_idx+1]
    tp[zero_z_end_idx][zero_z_end_idx + 1] = prob_z_branch / 2 #기존 iy 경로 확률 절반으로

    states += list(map(lambda x :{"word": "zero", "phone": "ih", "idx": x[0], "statePdf":x[1]}, enumerate(phone_hmm_data["ih"]["state_pdf_list"]))) # ih state 3개 추가
    tp[zero_z_end_idx][idx] = prob_z_branch / 2 #음소 시작은 ih 경로 확률과 같게
    tp[idx][idx] = phone_hmm_data["ih"]["tp"][1][1]
    tp[idx][idx + 1] = phone_hmm_data["ih"]["tp"][1][2]
    tp[idx + 1][idx + 1] = phone_hmm_data["ih"]["tp"][2][2]
    tp[idx + 1][idx + 2] = phone_hmm_data["ih"]["tp"][2][3]
    tp[idx + 2][idx + 2] = phone_hmm_data["ih"]["tp"][3][3]
    tp[idx + 2][zero_r_begin_idx] = phone_hmm_data["ih"]["tp"][3][4]

    for (str1, str2), prob in bigram.items():
        last = word_end_idx_dict[str1]
        if not str1 == "<s>":
            tp[last-1][word_begin_idx_dict[str2]] = LANGUAGE_MODEL_WEIGHT * tp[last-1][last] * last_sp_skip_prob[str1] * prob
        tp[last][word_begin_idx_dict[str2]] = LANGUAGE_MODEL_WEIGHT * last_sp_end_prob[str1] * prob

    # Make word_idx_info (word_begin_idx_dic = {idx: "word"}, word_end_idx_list = [0,11,...])
    begin_idx_to_word = dict([(value, key) for key, value in word_begin_idx_dict.items()])
    word_to_end_idx = word_end_idx_dict
    # print(word_begin_idx_dict)
    # print(word_end_idx_dict)
    word_idx_info = (begin_idx_to_word, word_to_end_idx)

    return word_idx_info, Hmm(prior_prob, tp, states)


#TODO 3. 비터비 알고리즘 이용해 모든 테스트 파일(tst.zip)에 대해 best state sequence 구하고 단어 sequence로 만든 다음 결과 출력

def viterbi(word_idx_info, evidence, hmm):
    #Viterbi algorithm for given MFCC vector file

    NUM_OF_STATES = hmm.NUM_OF_STATES
    m1 = np.empty((len(evidence),hmm.NUM_OF_STATES))
    m2 = np.empty((len(evidence),hmm.NUM_OF_STATES),dtype='i8')
    m1[0] = hmm.logscale_prior_prob + hmm.log_sensor_arr(evidence[0]) #statewise logsum, sensor: state마다 log(p(e|s))를 계산한 배열
    m2[0] = np.zeros(NUM_OF_STATES)
    m2.fill(-1) #이게 문제? TODO

    for t in range(1, len(evidence)):
        if t % 50 == 0:
            print(f"\tProcessing MFCC vector {t}/{len(evidence)-1}")
        m1[t] = np.max(m1[t-1] + np.transpose(hmm.logscale_tp), axis=1) + hmm.log_sensor_arr(evidence[t]) #m1[t]: 시간 t에서 각 state로 올 수 있는 가장 큰 확률 * 현재 state의 sensor
        m2[t] = np.argmax(m1[t-1] + np.transpose(hmm.logscale_tp), axis=1)

    q = np.empty(len(evidence),dtype='i8')
    q[len(evidence)-1] = np.argmax(m1[len(evidence)-1])
    for t in range(len(evidence)-2,-1,-1):
        q[t] = m2[t][q[t+1]]
    print(f"\tBest state sequence: \n{q}")


    #Get word sequence from best state sequence

    word_seq = []
    cur_word = ""
    begin_idx_to_word, word_to_end_idx = word_idx_info
    state = -1
    last_state = -1
    for state in q[1:]:
        if state in begin_idx_to_word:
            cur_word = begin_idx_to_word[state]
        if state == word_to_end_idx[cur_word]-1 and last_state != state: #word_begin_idx_dict = {98: "seven"}
            if cur_word != "<s>":
                word_seq.append(cur_word)
        last_state = state

    #절반 이상 탐색했으면 단어로?
    #<s>는 출력 안함
    print(f"\tComplete")
    if word_seq[0] == "two":
        return word_seq[1:]
    return word_seq

if __name__ == "__main__":

    begin = time.time()

    unigram, bigram, dictionary, phone_hmm_data = get_data()
    word_idx_info, word_seq_hmm = construct_word_seq_hmm(unigram, bigram, dictionary, phone_hmm_data)

    test_data = get_test_data()
    result_data = OrderedDict()

    with open(f'{FILEPATH_RECOGNIZED}/recognized.txt','w') as recognized_file:
        recognized_file.write("#!MLF!#\n")
        num_file = len(test_data)
        for i, (name, MFCC_vec_array) in enumerate(test_data.items()):
            begin_file = time.time()
            print(f"\n\nProcessing {name}.. [{i}/{num_file}]")
            word_seq = viterbi(word_idx_info, MFCC_vec_array, word_seq_hmm)
            print(f"Word seq: {word_seq}")
            recognized_file.write(f"{name.split('.')[0] + '.rec'}\n")
            for word in word_seq:
                recognized_file.write(f"{word}\n")
            recognized_file.write(".\n")
            print(f"소요 시간: {'{0:0.2f}'.format(time.time() - begin_file)}초")

    print(f"총 소요 시간: {'{0:0.2f}'.format(time.time() - begin)}초")


