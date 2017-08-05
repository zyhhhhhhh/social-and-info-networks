import numpy as np


class perfect_circles:
    """
    Dummy Model that always returns the ground truth circles
    """

    def __init__(self):
        pass

    def train(self,networks):
        """

        :param networks: list of egonet objects
        :return: Predictions on training networks
        """

        return self.predict(networks)

    def predict(self,networks):
        """

        :param networks:
        :return:
        """
        # Normally some training would take place here...


        return [n.circles for n in networks]



class random_circles:
    """
    Dummy model
    """

    def __init__(self,n_cricles=2,max_nodes_per_circle=10):
        self.n_circles = n_cricles
        self.max_nodes_per_circle = max_nodes_per_circle

    def train(self,networks):
        """
        No training actually takes place in this class.
        :param networks: list of egonet objects
        :return: Predictions on training networks
        """

        # Normally some training would take place here...

        return self.predict(networks)



    def predict(self,networks):
        """
        Randomly constract valid circles given initialized hyper parameters
        :param networks:
        :return:
        """

        predictions = []
        for n in networks:

            #if not isinstance(n,egonet):
            #    raise TypeError('Given a non egonet item')

            predicted_circles = {}
            for c in range(self.n_circles):
                predicted_circles[c] = np.unique(np.random.choice(n.node_ids,
                                                                  min(self.max_nodes_per_circle,n.n_nodes),
                                                                  replace=False)).tolist()
            predictions.append(predicted_circles)

        return predictions



