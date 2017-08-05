from os.path import isdir
from glob import glob
from Main.EgoNet import read_circles

class CESNA:
    """
    CESNA Model
    """

    def __init__(self,pretrained_path):
        """
        If pretrained circles already exist, you can supply a path and the training phase will be skipped
        :param pretrained_path:
        """
        self.pretrained_path = pretrained_path
        pass

    def train(self,networks):
        """

        :param networks: list of egonet objects
        :return: Predictions on training networks
        """
        #if isdir(self.pretrained_path):
        #    print('Pretraining Detected!\nSkipping Training Phase')

        return self.predict(networks)

    def predict(self,networks):
        """

        :param networks:
        :return:
        """
        if isdir(self.pretrained_path):
            # make dict mapper
            files = glob(self.pretrained_path + '/*.circles')
            mapper = {}
            for file in files:
                id = int(file.split('/')[-1].strip('.cricles'))
                mapper[id] = file

        found_circles = []
        for n in networks:
            found_circles.append(read_circles(mapper[int(n.egoid)]))

        if len(found_circles) != len(networks):
            print('\n\n\nOH NO, length mismatch!!!!\n\n\n')


        return found_circles
