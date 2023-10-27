from .base import BaseGenerator, BaseDataset
import numpy as np

class UrnGenerator(BaseGenerator):
    """
    Generates urn examples.  The base case is given by the templates.
    """
    def __init__(self, colors: np.ndarray, rng: np.random.default_rng) -> None:
        super(UrnGenerator, self).__init__(rng=rng)
        self.prompt =("An urn is placed in front of you. It has [list_balls] balls. "
                      "A ball is drawn at random from the urn.\n")
        self.instruction =("Fill in the [BLANK] with your answer and just your answer.\n"
                           "If the answer is a number, it should be a decimal number like 0.10.\n"
                           "Round up and round down your answer to no more than 2 decimal places.\n"
                           "For example, QUESTION: The chance the ball is blue is [BLANK].\n"
                           "YOU: 0.25\n")
        self.template1 = "The ball is [color]."
        self.template2 = "The chance the ball is [color] is [BLANK]."
        self.template3 = "The color most likely to be drawn is [BLANK]."
        self.template4 = "Between [color_i] balls and [color_j] balls, the most likely to be drawn is [BLANK]."
        self.colors = colors

        c_str = ""
        for i, c in enumerate(self.colors[:,0]):
            if i < self.colors.shape[0] - 1:
                c_str = c_str + self.colors[i,1] + " " + c + ", "
            else:
                c_str = c_str + 'and ' + self.colors[i,1] + " " + c
        
        self.prompt = self.prompt.replace("[list_balls]", c_str)
    
    def _gen_probe(self) -> str:
        """
        Returns a proble example.

        params:
        None

        returns:
        sample : str : the sample
        str(1) : str : 1 indicating that the value is true
        """
        indx = self.rng.integers(0, self.colors.shape[0])
        sample = self.template1.replace("[color]", self.colors[indx,0])
        return sample, str(1)
    
    def _gen_chance(self) -> tuple:
        """
        Returns a chance example.  The computed chance is just the ratio of the colored ball to 
        the total number of balls.

        params:
        None

        returns:
        sample : str : the sample with '[BLANK]'
        answer : str : the answer, rounded to 2 decimal places
        """
        indx = self.rng.integers(0, self.colors.shape[0])
        sample = self.template2.replace("[color]", self.colors[indx,0])
        answer = int(self.colors[indx,1]) / np.sum(self.colors[:,1].astype(int))
        answer = round(answer, 2)
        answer = str(answer)
        
        return sample, answer
    
    def _gen_likely(self) -> tuple:
        """
        Returns a sample that asks the most likely color of ball to be drawn.  Answer is computed as the 
        ball index with the largest number.  Should only be used if there are unequal balls types.

        params:
        None

        returns:
        sample : str : the sample
        answer : str : the color of the ball with the highest chance of being drawn
        """
        indx = np.argmax(self.colors[:,1].astype(int))
        answer = self.colors[indx,0]

        return self.template3, answer
    
    def _gen_between(self) -> tuple:
        """
        Creates a sample for questions about the likelihood of two different balls to be drawn.
        The answer is computed by just comparing how many balls of the two types chosen there are.

        params:
        None

        returns:
        sample: str : the sample
        answer : str : the color of the ball that is more likely to be drawn
        """
        # Choose two different random indices
        indx1 = self.rng.integers(0, self.colors.shape[0])
        indx2 = indx1
        while indx1 == indx2:
            indx2 = self.rng.integers(0, self.colors.shape[0])
        
        # Build the sample
        sample = self.template4.replace("[color_i]", self.colors[indx1,0])
        sample = sample.replace("[color_j]", self.colors[indx2,0])

        # Get the answer
        answer = indx1 if int(self.colors[indx1,1]) >= int(self.colors[indx2,1]) else indx2
        answer = self.colors[answer,0]

        return sample, answer

    def gen_base_sample(self, task: int) -> tuple:
        if task > 3:
            raise ValueError("Your task value exceeded the maximum number of tasks.")
        
        if task == 0:
            sample, answer = self._gen_probe()
        elif task == 1:
            sample, answer = self._gen_chance()
        elif task == 2:
            sample, answer = self._gen_likely()
        else:
            sample, answer = self._gen_between()
        
        return sample, answer
    
    def generate_samples(self, ns: list, tasks: list, prompt=False, unique=False) -> np.ndarray:
        """
        Generates a set of base samples.

        params:
        ns : list : list of integers that specify the number of samples to be created for each task
        tasks : list : list of integer for the tasks to be done for each ns
        unique : bool : specify whether the output should only have unique samples

        returns:
        data : np.ndarray : numpy array of the samples
        """

        # Check whether ns and tasks lists are the same size
        if len(ns) != len(tasks):
            raise ValueError("ns and tasks inputs not the same length.")
        
        samples = []
        ans = []

        for n, t in zip(ns, tasks):
            for i in range(n):
                s, a = self.gen_base_sample(t)

                if prompt and t < 1:
                    s = self.prompt + s
                elif prompt and t > 0:
                    s = self.prompt + self.instruction + "QUESTION: " + s

                samples.append(s)
                ans.append(a)
        
        data = self.cat_xy(samples, ans)

        if unique:
            data = np.unique(data, axis=0)
        
        return data