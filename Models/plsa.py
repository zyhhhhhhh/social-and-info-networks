from __future__ import division
import matplotlib
matplotlib.use('Agg')
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
import tempfile

class plsa_wrapper:
    def __init__(self):
        pass

    def train(self,networks):
        """
        Each network circle set is independently learned using a plsa model
        :param networks: list of egonet objects
        :return: Predictions on training networks
        """
        return self.predict(networks)

    def predict(self,networks):
        """
        Predict Circles by looking at topic probabilities
        :param networks:
        :return:
        """

        network_predictions = []
        for network in networks:

            # Get edge list format from egonet
            ep = network.edges_path
            X_feat = network.features.values
            print(network.features)



            tmpdir = tempfile.mkdtemp()
            doc = tmpdir + '/document.txt'


            # Convert every feature set into a document format
            index_2_id = {}
            with open(doc,'w') as document:
                for index,node_id in enumerate(network.node_ids):
                    index_2_id[index] = node_id
                    df = network.features[network.features['nodeid'] == node_id]
                    for val,feature in zip(df.values[0],df.columns.values):
                        if val > 0:
                            document.write(feature+' ')
                    document.write('\n')

            network_plsa = plsa(file=doc)
            network_plsa.learn(max_iter=50)
            print('\n\ntop words\n\n')
            print(network_plsa.get_topwords())
            circles = network_plsa.get_clusters()
            circles = {key:[index_2_id[index] for index in circles[key]] for key in circles}
            print(circles)
            network_predictions.append(circles)

        return network_predictions






class plsa:

    def __init__(self,file,background_prob_lambda=0.5,K_topics=10,seed=1):
        """
        Initalize data structures for learning a PLSA model over the document collection defined
        word and document nomencalture used because that is what is most familiar to general audiences
        in this context, every word is a feature and every document is a node in the ego network

        :param file: file to use
        :param background_prob_lambda: the probability that a word is drawn from the background distribution
        :param K_topics: the number of topics in the model
        :param seed: int controlling the random initialization process
        """


        print("\n\nInitializing PLSA with Background Model:")

        self.seed = seed
        np.random.seed(self.seed)
        self.file = file
        self.Z = background_prob_lambda
        self.K = K_topics
        self.loglikelihoods = []
        self.rc_loglikelihoods = []

        # Init Data
        self.background = defaultdict(float)
        self.document = []
        self.num_words = 0
        self.num_docs = 0

        with open(file) as f:
            for doc in f:
                d = doc.strip('\n').split()
                self.num_docs += 1
                words = defaultdict(int)
                for w in d:
                    self.num_words += 1
                    self.background[w] += 1
                    words[w] += 1
                self.document.append(words)
            for w in self.background:
                self.background[w] = self.background[w] / self.num_words

        self.vocab_size = len(self.background)

        # Init Model Params
        self.topics_word = {}
        self.probs_per_topic = []

        for index in range(0,self.K):
            probs = np.random.uniform(.33,.67, size=self.vocab_size)
            self.probs_per_topic.append(probs/np.sum(probs))

        for index,word in enumerate(self.background):
            self.topics_word[word] = []
            for topic in range(0,self.K):
                self.topics_word[word].append(self.probs_per_topic[topic][index])

        self.topics_doc = []
        for index in range(0,self.num_docs):
            probs = np.random.uniform(.33,.67, size=self.K)
            self.topics_doc.append(probs/np.sum(probs))

        # Compute initial log-likelihood
        self.get_loglikelihood()
        self.loglikelihoods.append(self.loglikelihood)

        print('# Topics: %d' % self.K)
        print('P(Background|w): %f' % self.Z)
        print('# Documents: %d' % self.num_docs)
        print('# Words: %d' % self.num_words)
        print('Vocab Size: %d' % self.vocab_size)
        print('Initial Log-Likelihood: %.5f'%self.loglikelihood)

    def get_clusters(self,threshold = 0.001):
        """
        Make a circles dictionary , once cirlce for every topic
        :param threshold:
        :return:
        """
        circles = defaultdict(list)
        for j,topic_vector in enumerate(self.topics_doc):
            #print(j)
            for i,prob in enumerate(topic_vector):
                #print(i)
                if prob > threshold:
                    circles[i].append(j)

        return circles




    def get_loglikelihood(self,offset = 1E-7):
        """
        Compute the log likelihood of the data given the model params
        :return:
        """
        self.loglikelihood = 0
        for index, document in enumerate(self.document):
            for word in document:
                count = document[word]
                bg = self.background[word]
                word_topic = np.array(self.topics_word[word])
                doc_topic = np.array(self.topics_doc[index])
                dot = max(np.dot(word_topic, doc_topic), offset)
                self.loglikelihood += count * np.log(self.Z * bg + (1 - self.Z) * dot)


    def EM(self,offset = 1E-7):
        """
        Perform a single STEP of Expectation maximization
        :return:
        """
        self.loglikelihood = 0
        new_topics_doc = []
        new_topics_word = defaultdict(float)
        tw = np.zeros(self.K)
        for index,document in enumerate(tqdm(self.document,desc="EM Step %d"%self.step,leave=True,miniters=100)):
            td = np.zeros(self.K)
            for word in document:
                count = document[word]
                bg = self.background[word]
                word_topic = np.array(self.topics_word[word])
                doc_topic = np.array(self.topics_doc[index])
                dot = max(np.dot(word_topic,doc_topic),offset)

                self.loglikelihood += count*np.log(self.Z*bg+(1-self.Z)*dot)

                pj = word_topic*doc_topic/dot
                pB = self.Z*bg/(self.Z*bg+(1-self.Z)*dot)

                M = count*(1-pB)*pj
                td = td+M

                new_topics_word[word] += M

            # Out of word Loop
            new_topics_doc.append(td/np.sum(td))
            tw = tw + td

        # Normalize
        for word in new_topics_word:
            new_topics_word[word] /= tw

        # Update old with new
        self.topics_word = new_topics_word
        self.topics_doc = new_topics_doc

    def get_topwords(self,n_words=10):
        """
        print out the top n words and write a file of the top n_words for each topic in the model
        :param n_words:
        :return:
        """
        i2w = {i: w for i, w in enumerate(self.topics_word)}
        w2i = {i2w[i]: i for i in range(len(self.topics_word))}

        d = np.array([self.topics_word[i2w[i]] for i in range(len(self.topics_word))])
        n_docs, n_topics = d.shape
        with open("TopicWords%f.txt"%(self.Z*10),mode='w') as f:
            for k in range(n_topics):
                string = 'Topic %d: %s' % (k, ' '.join([i2w[i] for i in np.argsort(d[:, k])[-n_words:]]))
                print(string)
                f.write(string+'\n')
        return True

    def learn(self,max_iter=100,threshold=0.0001):
        """
        Perform EM update steps until max iter is reached or until convergence given a threshold is reached

        :param max_iter: maximum # of iterations of EM to perform
        :param threshold: Stopping criterion for relative change in loglikelihood
        :return:
        """

        for step in range(max_iter):
            self.step = step
            self.EM()
            self.loglikelihoods.append(self.loglikelihood)
            rcl = (self.loglikelihoods[-1] - self.loglikelihoods[-2]) / self.loglikelihoods[-2] * (-1)
            self.rc_loglikelihoods.append(rcl)
            print('Log-Likelihood: %.5f  Relative Change in Log-Likelihood: %.5f' % (self.loglikelihoods[-1], self.rc_loglikelihoods[-1]))

            if rcl < threshold and step != 0:
                print("Converged!")
                break


    def plot(self,ll_path='loglikelihood.png',rcll_path='RC_loglikelihood'):
        """
        Make plots of the model likelihood convergence and the rate of change for that convergence
        :return:
        """
        plt.figure()
        plt.plot(range(len(self.loglikelihoods[1:])),self.loglikelihoods[1:])
        plt.title('Log-Likelihood Convergence\n'
                  ' Background Topic Prob:%f \t # of Topics:%d'%(self.Z,self.K))
        plt.xlabel('Iteration #')
        plt.ylabel('Log-Likelihood')
        sns.despine(trim=True)
        plt.savefig(ll_path)

        plt.figure()
        plt.plot(range(len(self.rc_loglikelihoods[1:])), self.rc_loglikelihoods[1:])
        plt.title('Rate of Change in Log-Likelihood Convergence\n'
                  ' Background Topic Prob:%f \t # of Topics:%d' % (self.Z, self.K))
        plt.xlabel('Iteration #')
        plt.ylabel('Relative Change in Log-Likelihood')
        sns.despine(trim=True)
        plt.savefig(rcll_path)


