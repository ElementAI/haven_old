class ExpManager:
    def __init__(self, exp_dict, savedir_base):
        self.exp_dict = exp_dict
        self.savedir_base = savedir_base
        self.score_list = []
    
    def create_checkpoint(self, reset):
        """
        creating the folder for the experiment
        """
        pass
    
    def save_checkpoint(self, state_dict):
        """
        saving the experiment
        """
        pass

    def load_checkpoint(self):
        """
        loading the experiment
        """
        return state_dict

    def add_score_dict(self, score_dict):
        """
        add score dict to score_list
        """
        self.score_list += [score_dict]

    def create_jupyter(self, reset):
        """
        creating the folder for the experiment
        """
        pass

    def launch_jupyter(self, reset):
        """
        creating the folder for the experiment
        """
        pass