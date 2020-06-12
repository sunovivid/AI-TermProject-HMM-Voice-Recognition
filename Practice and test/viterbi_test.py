import numpy as np

class Hmm():
    def __init__(self, prior_prob, tp, states):
        self.logscale_prior_prob = np.log(prior_prob)
        self.logscale_tp = np.log(tp)
        self.states = states

    @property
    def NUM_OF_STATES(self):
        return len(self.states) #state idx = 0 to n(116)이나, states에 시작 state는 포함되어 있지 않으므로 +1

    def log_sensor_arr(self, evidence):
        if evidence == 1:
            return np.log(np.array([.2, .5, .0]))
        elif evidence == 2:
            return np.log(np.array([.4, .4, .0]))
        elif evidence == 3:
            return np.log(np.array([.4, .1, .0]))

def pprint(arr):
    print("type:{}".format(type(arr)))
    print("shape: {}, dimension: {}, dtype:{}".format(arr.shape, arr.ndim, arr.dtype))
    print("Array's Data:\n", arr)

def viterbi(evidence, hmm):
    # 주어지는 observation이 MFCC벡터 차원이므로 모든 hmm이 MFCC 벡터 수준이어야 함
    NUM_OF_STATES = hmm.NUM_OF_STATES
    m1 = np.empty((len(evidence), hmm.NUM_OF_STATES))
    m2 = np.empty((len(evidence), hmm.NUM_OF_STATES), dtype='i8')
    m1[0] = hmm.logscale_prior_prob + hmm.log_sensor_arr(evidence[0])
    m2[0] = np.zeros(NUM_OF_STATES)
    t=0
    print(f"m1[{t}]: {m1[t]}")
    print(f"m2[{t}]: {m1[t]}")

    for t in range(1, len(evidence)):
        m1[t] = np.max(m1[t - 1] + np.transpose(hmm.logscale_tp), axis=1) + hmm.log_sensor_arr(evidence[t])  # m1[t]: 시간 t에서 각 state로 올 수 있는 가장 큰 확률 * 현재 state의 sensor
        m2[t] = np.argmax(m1[t - 1] + np.transpose(hmm.logscale_tp), axis=1)

        print()
        print(m1[t - 1] + np.transpose(hmm.logscale_tp))
        print(np.max(m1[t - 1] + np.transpose(hmm.logscale_tp), axis=1))
        print(f"m1[{t}]: {m1[t]}")
        print(f"m2[{t}]: {m2[t]}")

    q = np.empty(len(evidence), dtype='i8')
    q[len(evidence) - 1] = np.argmax(m1[len(evidence) - 1])
    print(np.argmax(m1[len(evidence) - 1]))

    for t in range(len(evidence) - 2, -1, -1):
        q[t] = m2[t][q[t + 1]]

    return q


def test():
    prior_prob = np.array([0.8,0.2,0])
    tp = np.array([[0.6,0.3,0.1],
                   [0.4,0.5,0.1],
                   [0.0,0.0,0.0]])
    states = {"HOT","COLD", "END"}
    hmm = Hmm(prior_prob, tp, states)

    evidence = [3,1,3,3,1]

    print(viterbi(evidence, hmm))

test()