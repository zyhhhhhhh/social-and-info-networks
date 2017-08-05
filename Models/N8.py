import numpy as np
from utils.datapath import DataPath
from Models.Louvain import Louvain
from collections import defaultdict, Counter, OrderedDict
import os
import glob
from Main.EgoNet import dirsep, egonet
from warnings import warn
from pprint import pprint
import pickle
from Main.Evaluation import Train_Test_Splitter, Evaluate, plot_scores
import tempfile
import subprocess


class LINELDA():
    """
    Model First learns vector space model of network
    Then performs LDA over the concatenated space of the features and Network Vectors
    Then based on threshold, includes nodes in a circle,
    Uses held out data to pick best
    """

    def __init__(self):
        """
        """
        self.line = node2vec(n_componenets=3,walk_length=10,num_walks=100,window_size=10,n_iter=10) # tsvd_embed(50)



        pass

    def train(self,networks):
        """
        Each network circle set is independently learned
        :param networks: list of egonet objects
        :return: Predictions on training networks
        """
        return self.predict(networks)

    def predict(self,networks):
        """
        Predict Circles by leveraging louvain on a modified network
        :param networks:
        :return:
        """



        for network in networks:

            # Get edge list format from egonet
            ep = network.edges_path
            X_emb = self.line.fit_transform(ep)
            X_feat = network.features.values()
            print(X_emb.shape)
            print(X_feat.shape)





        return [n.circles for n in networks]


class node2vec:
    def __init__(self,n_componenets=100,walk_length=10,num_walks=10,window_size=10,n_iter=1,workers=8,p=1,q=1,weighted=True,undirected=True):
        self.ncomps = n_componenets
        self.walklength = walk_length
        self.numwalks = num_walks
        self.windowsize = window_size
        self.iter = n_iter
        self.workers = workers
        self.p = p
        self.q = q
        self.weighted  = weighted
        self.undirected = undirected

        if self.weighted == True:
            self.weighted_string = '--weighted'
        else:
            self.weighted_string = '--unweighted'

        if self.undirected == True:
            self.directed_string = '--undirected'
        else:
            self.directed_string = '--directed'


    def fit_transform(self,in_file):
        """

        :param G:
        :return:
        """
        tmpdir = tempfile.mkdtemp()
        out_file = tmpdir + '/outfile.txt'

        id_to_index = {}
        index_2_id = {}



        counter = 0
        with open(in_file,'r') as inf:
            with open(tmpdir + '/infile.txt','w') as outf:
                for line in inf:
                    a,b = line.split(' ')
                    if a in id_to_index:
                        pass
                    else:
                        id_to_index[a] = counter
                        index_2_id[counter] = a
                        counter += 1
                    if b in id_to_index:
                        pass
                    else:
                        id_to_index[b] = counter
                        index_2_id[counter] = b
                        counter += 1

                    outf.write(line.strip('\n')+' 1\n')
                    print(line.strip('\n')+' 1\n')
        in_file = tmpdir + '/infile.txt'





        start_dir = os.curdir

        print("Starting Dir: " + start_dir)

        current_pardir = os.path.abspath(os.path.join(os.curdir, os.pardir))
        os.chdir(os.path.join(os.path.dirname(__file__), 'node2vec/src'))

        arg_list = ['python','main.py',
                    '--dimensions', str(self.ncomps),
                    '--output',out_file,
                    '--input',in_file,
                    '--walk-length', str(self.walklength),
                    '--num-walks', str(self.numwalks),
                    '--window-size', str(self.windowsize),
                    '--iter', str(self.iter),
                    '--workers', str(self.workers),
                    '--p', str(self.p),
                    '--q', str(self.q),
                    self.weighted_string,
                    self.directed_string,
                    ]
        print(' '.join(arg_list))
        subprocess.call(arg_list)
        # Change back to origanl dir
        os.chdir(current_pardir + '/Models/')
        print("Ending Dir: " + os.curdir)

        # Read result back into memory
        print("\nReading LINE output vectors into Python Memory...")
        D = embedding_2_X(out_file)
        print("Done")

        return D


def embedding_2_X(path):
    """

    :param path:
    :return:
    """
    counter = -1
    with open(path, mode='r') as f:
        for line in f:

            if counter == -1:
                n, p = line.split()
                n, p = int(n), int(p)

                print('Output dimensions', n, p)
                X = np.ones((n, p))
                labels = np.ones(n)

            else:
                row_id = int(line.split()[0])
                X[row_id, 0:p] = np.array([float(x) for x in line.split()[1:p + 1]])

            counter += 1

    print("\nData Shape:" + str(X.shape))

    return X


class LINE:
    def __init__(self,n_components=100,order=2,
                 neg=5,nthreads=8,nsamples=10,depth=2,kmax=500,threshold=5):
        self.n_components = n_components
        self.order = order
        self.neg = neg
        self.nthreads = nthreads
        self.nsamples = nsamples
        self.depth = depth
        self.kmax = kmax
        self.threshold = threshold


    def read_line_vector_file(self,path):
        """

        :param path:
        :return:
        """
        counter = -1
        with open(path, mode='r') as f:
            for line in f:

                if counter == -1:
                    n, p = line.split()
                    n, p = int(n), int(p)

                    print('Output dimensions',n,p)
                    X = np.ones((n, p))
                    labels = np.ones(n)

                else:
                    row_id = int(line.split()[0])
                    X[row_id, 0:p] = np.array([float(x) for x in line.split()[1:p + 1]])

                counter += 1

        print("\nData Shape:" + str(X.shape))

        return X

    def fit_transform(self,in_file):
        """

        :param G:
        :return:
        """

        tmpdir = tempfile.mkdtemp()
        out_file = tmpdir+'/outfile.txt'

        start_dir = os.curdir

        print("Starting Dir: "+start_dir)

        current_pardir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        os.chdir(current_pardir+'/Models/LINE/linux/')
        print('Mod Dir: %s'%os.curdir)
        arg_list = [
                     './run_line.sh',
                     '-d',str(self.n_components),
                     '-o', str(self.order),
                     '-i', str(in_file),
                     '-x', str(out_file),
                     '-t', str(self.nthreads),
                     '-n', str(self.neg),
                     '-s', str(self.nsamples),
                     '-k', str(self.kmax),
                     '-p', str(self.depth),
                     '-r',str(self.threshold),
                     ]
        print(' '.join(arg_list))
        # Change to dir and call bash with arg list
        os.chdir(os.path.join(os.path.dirname(__file__),'LINE/linux/'))
        subprocess.call(arg_list,shell=True)
        # Change back to origanl dir
        os.chdir(os.path.dirname(__file__))
        print("Ending Dir: " + os.curdir)

        # Read result back into memory
        print("\nReading LINE output vectors into Python Memory...")
        D = self.read_line_vector_file(out_file)
        print("Done")

        return D








class some_model:
    """
    Dummy Model that always returns the ground truth circles
    """

    def __init__(self,datapath):
        """

        """
        self.datapath = datapath
        self.lv = Louvain()
        pass

    def train(self,networks):
        """
        Train a model to predict co-membership of networks given features and network structure
        :param networks: list of egonet objects
        :return: Predictions on training networks
        """
        print('Some Model Traing Phase Started')

        # Get Feature Mapper
        print('Getting Feature Mapper for network source: ')
        network_source = networks[0].network_source
        print(network_source)
        self.feature_mapper = pickle.load(open(os.path.join(datapath.features, "%s" % network_source), "rb"))
        self.n_features = len(self.feature_mapper)

        # Collect Features Matrix

        # Build Prediction Data Set

        # Train Predictive Model

        # Scan P threshold



        return self.predict(networks)


    def network_2_features(self,network):
        """
        Builds feature representation of a network
        :param network:
        :return:
        """
        self.feature_mapper
        X = {nodeid:np.zeros(self.n_features) for nodeid in network.node_ids}
        df = network.features
        for nodeid,col in enumerate(df):
            pass



        return X



    def pred_comembership(self,a,b):
        """
        Predict if two edges are in a group with each other
        :param a: network a
        :param b: network b
        :return: probability
        """

        return np.random.rand()[0]


    def predict(self,networks):
        """
        Predict Circles by leveraging louvain on a modified network
        :param networks:
        :return:
        """
        # Normally some training would take place here...

        for network in networks:
            # Replace egonetwork edge set with new predicted comemebership edge structure
            network.G = 1




        return [n.circles for n in networks]





if __name__ == '__main__':

    # Params
    max_nets_per_source = 200

    # Get Data Paths
    datapath = DataPath(verbose=False)
    data_paths = {'facebook': datapath.facebook,
                  'twitter': datapath.twitter,
                  'gplus': datapath.gplus}

        # Pre Build Features
    for network_source in data_paths:

        DATA_PATH = data_paths[network_source]
        train_test = defaultdict(list)


        assert isinstance(DATA_PATH,str)
        if os.path.isdir(DATA_PATH):
            files = glob.glob(DATA_PATH+dirsep+'*')
            node_ids = {}
            for f in files:
                nodeid, ftype = os.path.basename(f).split('.')
                node_ids[nodeid] = nodeid


            source_features = []

            networks = []
            for nodeid in node_ids.keys()[0:min(max_nets_per_source,len(node_ids))]:
                try:
                    e = egonet(dir=DATA_PATH,egoid=nodeid,network_source=network_source,verbose=True)

                    print(type(e.feature_names))
                    source_features.extend(e.feature_names.values())


                except ValueError as ve:
                    warn(ve.message)

            source_feature_counter = Counter(source_features)
            source_features_dict = dict(zip(source_feature_counter.keys(),range(len(source_feature_counter))))
            pickle.dump(source_features_dict, open(os.path.join(datapath.features,"%s"%network_source), "wb"))
            print('\n\n\n\n')
            print(len(source_features_dict))
            pprint(source_feature_counter)
            print('\n\n\n\n')


        else:
            warn('')

        # Declare Methods
        methods = OrderedDict((('Some Model', some_model(datapath=datapath)),
                               ('Louvain', Louvain()),
                               ))

        # Build Training and Testing Dicts
        dsd = Train_Test_Splitter(data_paths, alpha=0.1, max_nets_per_source=3)

        # Perform Evaluation
        sdf = Evaluate(methods, dsd)
        plot_scores(sdf, datapath.results + '/N8_')




