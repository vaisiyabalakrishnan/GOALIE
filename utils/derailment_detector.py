class DerailmentDetector:
    def __init__(self, max_repeats=3):
        self.last_actions = []
        self.last_obs = ""
        self.max_repeats = max_repeats
        self.derailment_reason = ""


    def update(self, action):
        self.last_actions.append(action)
        if len(self.last_actions) > self.max_repeats:
            self.last_actions.pop(0)


    def is_derailed(self):
        if len(set(self.last_actions)) == 1 and len(self.last_actions) == self.max_repeats:
            self.derailment_reason = "Your last 3 actions were the same."
            print("\n [DERAILMENT DETECTOR] Detected 3 repeated steps.\n")
            return True
        if "nothing happens" in self.last_obs.lower():
            self.derailment_reason = f"Your last action: {self.last_actions[-1]} resulted in 'nothing happens' in the last observation."
            print("\n [DERAILMENT DETECTOR] Detected 'nothing happens' in the last observation.\n")
            return True
        return False