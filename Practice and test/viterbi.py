import numpy as np

# #state 0 = rainy, state 1 = sunny
# transition_prob = np.array([
#     [0.7, 0.3],
#     [0.3, 0.7]
# ])
#
# #state rainy => 우산 가져올 확률 0.9
# #      sunny => 우산 가져올 확률 0.2
# emission_prob = np.array([
#     [0.9, 0],
#     [0, 0.2]
# ]) #rainny state의 emission prob

obs = ('normal', 'cold', 'dizzy')
states = ('Healthy', 'Fever')
start_p = {'Healthy': 0.6, 'Fever': 0.4}
trans_p = {
 'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
 'Fever': {'Healthy': 0.4, 'Fever': 0.6}
}
emit_p = {
 'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
 'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
}

def viterbi(obs, states, start_prob, transition_prob, emission_prob):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_prob[st] * emission_prob[st][obs[0]], "prev":None}
        for t in range(1, len(obs)):
            V.append({})
            for st in states:
                max_tr_prob = max(V[t-1][prev_st]["prob"] * transition_prob[prev_st][st] for prev_st in states) #
                for prev_st in states:
                    if V[t-1][prev_st]["prob"] * transition_prob[prev_st][st] == max_tr_prob:
                        max_prob = max_tr_prob * emission_prob[st][obs[t]]
                        V[t][st] = {"prob":max_prob, "prev": prev_st}
                        break

        for line in dptable(V):
            print(line)

        opt = []
        max_prob = max(value["prob"] for value in V[-1].values())
        previous = None
        for st, data in V[-1].items():
            if data["prob"] == max_prob:
                opt.append(st)
                previous = st
                break

        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t+1][previous]["prev"])
            previous = V[t+1][previous]["prev"]

        print(f"The steps of states are {''.join(opt)} with highest prob of {max_prob}")

def dptable(V):
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield ".%7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)

viterbi(obs, states, start_p, trans_p, emit_p)