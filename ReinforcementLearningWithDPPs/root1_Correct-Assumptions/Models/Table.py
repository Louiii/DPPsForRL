import numpy as np

class Table_model:
    def __init__(self):
        self.step_apprx = {}

    def update(self, s, a, s_dash, r):
        if (s, a) in self.step_apprx:
            (old_ns, multiple) = self.step_apprx[(s,a)]

            if multiple:
                if s_dash not in old_ns:
                    old_ns.append(s_dash)
                    self.step_apprx[(s,a)] = (old_ns, True)
            else:
                if s_dash != old_ns:
                    self.step_apprx[(s,a)] = ([old_ns, s_dash], True)
        else:
            self.step_apprx[(s,a)] = (s_dash, False)

    def predict(self, s, a):
        if (s, a) in self.step_apprx:
            (old_ns, multiple) = self.step_apprx[(s,a)]
            if multiple:
                return old_ns[np.random.randint(len(old_ns))]
            return old_ns
        return s
        # ns = []
        # for (r, c), (dr, dc) in zip(s, a):
        #     ns.append((max(min(r + dr, 3), 0), max(min(c + dc, 6), 0)))
        # # print(set(ns))
        # return (tuple(ns), 0)