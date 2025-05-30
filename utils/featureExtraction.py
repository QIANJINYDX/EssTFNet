import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import sys
from Bio import SeqIO
import itertools
import os
from math import sqrt,pow
import os, pickle, itertools
import sys
import os,itertools
import pickle
from math import pow
elements="AGCT"
# 计算AGCT四个字母的个数
def getAsum(seq):
    return sum([1 for x in seq if x=="A"])
def getCsum(seq):
    return sum([1 for x in seq if x=="C"])
def getGsum(seq):
    return sum([1 for x in seq if x=="G"])
def getTsum(seq):
    return sum([1 for x in seq if x=="T"])

# 获得kmer词表
def get_kmer(k):
    res=[]
    global elements
    for combo in itertools.product(elements, repeat=k):
        res.append(''.join(combo))
    return res

#GC含量计算公式为：[（G+C的总数量）/（A+T+C+G的总数量）] * 100%
def get_feature_GCContent(seq):
    return [(getGsum(seq)+getCsum(seq))/len(seq)]

def get_feature_kmer(seq,k):
    kmer_list = []
    for i in range(len(seq)-k+1):
        kmer_list.append(seq[i:i+k])
    kmer_vector = get_kmer(k)
    return [kmer_list.count(v)/len(kmer_list) for v in kmer_vector]

def get_reverse_seq(seq):
    return seq[::-1].translate(str.maketrans("ATCG", "TAGC"))

def get_feature_zCurve(seq):
    """
    :param seq:
    :return: z曲线的x,y,z值
    """
    A=seq.count('A')
    T=seq.count('T')
    G=seq.count('G')
    C=seq.count('C')
    l=len(seq)
    return [(A+G-C-T)/l,(A+C-G-T)/l,(A+T-C-G)/l]

def get_feature_cumulativeSkew(seq):
    """
    计算序列的累积偏斜
    """
    A=seq.count('A')
    T=seq.count('T')
    G=seq.count('G')
    C=seq.count('C')
    return [(G-C)/(G+C), (A-T)/(A+T)]

def get_feature_atgcRatio(seq):
    """
    计算ATGC比例
    """
    return [(seq.count("A") + seq.count("T")) / (seq.count("G") + seq.count("C"))]

# def get_feature_pseknc(seq,k):
#     """
#     Get PseKNC features for a given sequence.
#     """
#     res=[]
#     for i in range(1,k+1):
#         v=list(itertools.product(elements,repeat=i))
#         for j in v:
#             res.append(seq.count(''.join(j)))
#     return res

def subSeq(seq,w,k):
    firstPhaseSubSeq=[]
    secondPhaseSubSeq=[]
    thirdPhaseSubSeq=[]
    PhaseSubSeq=[]
    #不同相位的索引
    if w!=1:
        firstPhaseIndex=range(0,len(seq),3)
        #print firstPhaseIndex
        secondPhaseIndex=range(1,len(seq),3)
        thirdPhaseIndex=range(2,len(seq),3)
        seqIndex=range(0,len(seq))
        for i1 in firstPhaseIndex:
            if w-2+i1+k in seqIndex:
                #print w-2+i1+k
                firstPhaseSubSeq.append(seq[i1:(i1+w-1)]+seq[w-2+i1+k])
        for i2 in secondPhaseIndex:
            if w-2+i2+k in seqIndex:
                
                secondPhaseSubSeq.append(seq[i2:(i2+w-1)]+seq[w-2+i2+k])
        for i3 in thirdPhaseIndex:
            #print thirdPhaseIndex
            if w-2+i3+k in seqIndex:
                thirdPhaseSubSeq.append(seq[i3:(i3+w-1)]+seq[w-2+i3+k])
                #print 'sss'
        PhaseSubSeq.append(firstPhaseSubSeq)
        PhaseSubSeq.append(secondPhaseSubSeq)
        PhaseSubSeq.append(thirdPhaseSubSeq)
        #print PhaseSubSeq
        return PhaseSubSeq
        
    else:
        firstPhaseIndex=range(0,len(seq),3)
        secondPhaseIndex=range(1,len(seq),3)
        thirdPhaseIndex=range(2,len(seq),3)
        for i1 in firstPhaseIndex:
            firstPhaseSubSeq.append(seq[i1])
        for i2 in secondPhaseIndex:
            secondPhaseSubSeq.append(seq[i2])
        for i3 in thirdPhaseIndex:
            thirdPhaseSubSeq.append(seq[i3])
        PhaseSubSeq.append(firstPhaseSubSeq)
        PhaseSubSeq.append(secondPhaseSubSeq)
        PhaseSubSeq.append(thirdPhaseSubSeq)
        return PhaseSubSeq
#生成整个序列的坐标    
def kwZcurve(seq,w,k):
    PhaseSubSeq=subSeq(seq,w,k)
    xf=[]
    yf=[]
    zf=[]
    seq2feature=[]
    for subPhase in PhaseSubSeq:
        x=RevisedX(subPhase,w)
        y=RevisedY(subPhase,w)
        z=RevisedZ(subPhase,w)
        # print(x,y,z)
        xf.extend(x)
        yf.extend(y)
        zf.extend(z)
    seq2feature.extend(xf)
    seq2feature.extend(yf)
    seq2feature.extend(zf)
    return seq2feature
    #print seq2feature

    
def RevisedX(baseList,w):
    base=['A','T','G','C']
    xValue=[]
    for combo in itertools.product(base, repeat=w-1):
        sequence = ''.join(combo)
        num1=float(baseList.count(sequence+'A'))+float(baseList.count(sequence+'G'))
        num2=float(baseList.count(sequence+'C'))+float(baseList.count(sequence+'T'))
        xValue.append((num1 - num2) / len(baseList))
    return xValue
def RevisedY(baseList,w):
    base=['A','T','G','C']
    yValue=[]
    for combo in itertools.product(base, repeat=w-1):
        sequence = ''.join(combo)
        num1=float(baseList.count(sequence+'A'))+float(baseList.count(sequence+'C'))
        num2=float(baseList.count(sequence+'G'))+float(baseList.count(sequence+'T'))
        yValue.append((num1 - num2) / len(baseList))
    return yValue
def RevisedZ(baseList,w):
    base=['A','T','G','C']
    zValue=[]
    for combo in itertools.product(base, repeat=w-1):
        sequence = ''.join(combo)
        num1=float(baseList.count(sequence+'A'))+float(baseList.count(sequence+'T'))
        num2=float(baseList.count(sequence+'G'))+float(baseList.count(sequence+'C'))
        zValue.append((num1 - num2) / len(baseList))
    return zValue

def lz_col_name(w,k,dir=1):
    """
    This function is used to generate the column names for the feature matrix.
    Parameters
    ----------
    w : int
    The window size.
    k : int
    The kmer size.
    dir : int
    # 序列的方向，1表示正向，-1表示反向。
    """
    Cols=[]
    base=['A','T','G','C']
    name=[]
    for i in range(3):
        if i==0:
            h="_first"
        elif i==1:
            h="_second"
        else:
            h="_third"
        for combo in itertools.product(base, repeat=w-1):
            sequence = ''.join(combo)+h
            if dir==1:
                name.append("Forward w:"+str(w)+"_k:"+str(k)+"_X_"+sequence)
            else:
                name.append("Backward w:"+str(w)+"_k:"+str(k)+"_X_"+sequence)
        for combo in itertools.product(base, repeat=w-1):
            sequence = ''.join(combo)+h
            if dir==1:
                name.append("Forward w:"+str(w)+"_k:"+str(k)+"_Y_"+sequence)
            else:
                name.append("Backward w:"+str(w)+"_k:"+str(k)+"_Y_"+sequence)
        for combo in itertools.product(base, repeat=w-1):
            sequence = ''.join(combo)+h
            if dir==1:
                name.append("Forward w:"+str(w)+"_k:"+str(k)+"_Z_"+sequence)
            else:
                name.append("Backward w:"+str(w)+"_k:"+str(k)+"_Z_"+sequence)
    Cols=Cols+name
    return Cols

def get_feature_lzcurver(seqs,w,k,dir,is_tqdm=True):
    if dir!=1:
        seqs=[get_reverse_seq(s) for s in seqs]
    feature=[]
    if is_tqdm:
        for s in tqdm(seqs):
            feature.append(kwZcurve(str(s),w,k))
    else:
        for s in seqs:
            feature.append(kwZcurve(str(s),w,k))
    return feature
# 构造PseKNC

def check_psenac(lamada, w, k):
    """Check the validation of parameter lamada, w and k.
    """
    try:
        if not isinstance(lamada, int) or lamada <= 0:
            raise ValueError("Error, parameter lamada must be an int type and larger than and equal to 0.")
        elif w > 1 or w < 0:
            raise ValueError("Error, parameter w must be ranged from 0 to 1.")
        elif not isinstance(k, int) or k <= 0:
            raise ValueError("Error, parameter k must be an int type and larger than 0.")
    except ValueError:
        raise


def get_sequence_list_and_phyche_value_psednc(input_data, extra_phyche_index=None):
    """For PseDNC, PseKNC, make sequence_list and phyche_value.

    :param input_data: file type or handle.
    :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                     the value is its physicochemical property value (list).
                               It means the user-defined physicochemical indices.
    """
    if extra_phyche_index is None:
        extra_phyche_index = {}

    original_phyche_value = {'AA': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11],
                             'AC': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04],
                             'AG': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62],
                             'AT': [1.07, 0.22, 0.62, -1.02, 2.51, 1.17],
                             'CA': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25],
                             'CC': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24],
                             'CG': [-1.66, -1.22, -0.44, -0.82, -0.29, -1.39],
                             'CT': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62],
                             'GA': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71],
                             'GC': [-0.08, 0.22, 1.33, -0.35, 0.65, 1.59],
                             'GG': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24],
                             'GT': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04],
                             'TA': [-1.23, -2.37, -0.44, -2.24, -1.51, -1.39],
                             'TC': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71],
                             'TG': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25],
                             'TT': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11]}

    sequence_list = get_data(input_data)
    phyche_value = extend_phyche_index(original_phyche_value, extra_phyche_index)

    return sequence_list, phyche_value


def get_sequence_list_and_phyche_value_pseknc(input_data, extra_phyche_index=None):
    """For PseDNC, PseKNC, make sequence_list and phyche_value.

    :param input_data: file type or handle.
    :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                     the value is its physicochemical property value (list).
                               It means the user-defined physicochemical indices.
    """
    if extra_phyche_index is None:
        extra_phyche_index = {}

    original_phyche_value = {
        'AA': [0.06, 0.5, 0.09, 1.59, 0.11, -0.11],
        'AC': [1.5, 0.5, 1.19, 0.13, 1.29, 1.04],
        'GT': [1.5, 0.5, 1.19, 0.13, 1.29, 1.04],
        'AG': [0.78, 0.36, -0.28, 0.68, -0.24, -0.62],
        'CC': [0.06, 1.08, -0.28, 0.56, -0.82, 0.24],
        'CA': [-1.38, -1.36, -1.01, -0.86, -0.62, -1.25],
        'CG': [-1.66, -1.22, -1.38, -0.82, -0.29, -1.39],
        'TT': [0.06, 0.5, 0.09, 1.59, 0.11, -0.11],
        'GG': [0.06, 1.08, -0.28, 0.56, -0.82, 0.24],
        'GC': [-0.08, 0.22, 2.3, -0.35, 0.65, 1.59],
        'AT': [1.07, 0.22, 0.83, -1.02, 2.51, 1.17],
        'GA': [-0.08, 0.5, 0.09, 0.13, -0.39, 0.71],
        'TG': [-1.38, -1.36, -1.01, -0.86, -0.62, -1.25],
        'TA': [-1.23, -2.37, -1.38, -2.24, -1.51, -1.39],
        'TC': [-0.08, 0.5, 0.09, 0.13, -0.39, 0.71],
        'CT': [0.78, 0.36, -0.28, 0.68, -0.24, -0.62]}

    sequence_list = get_data(input_data)
    phyche_value = extend_phyche_index(original_phyche_value, extra_phyche_index)

    return sequence_list, phyche_value


def get_sequence_list_and_phyche_value(input_data, k, phyche_index, extra_phyche_index, all_property):
    """For PseKNC-general make sequence_list and phyche_value.

    :param input_data: file type or handle.
    :param k: int, the value of k-tuple.
    :param k: physicochemical properties list.
    :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                     the value is its physicochemical property value (list).
                               It means the user-defined physicochemical indices.
    :param all_property: bool, choose all physicochemical properties or not.
    """
    if phyche_index is None:
        phyche_index = []
    if extra_phyche_index is None:
        extra_phyche_index = {}

    diphyche_list = ['Base stacking', 'Protein induced deformability', 'B-DNA twist', 'Dinucleotide GC Content',
                     'A-philicity', 'Propeller twist', 'Duplex stability:(freeenergy)',
                     'Duplex tability(disruptenergy)', 'DNA denaturation', 'Bending stiffness', 'Protein DNA twist',
                     'Stabilising energy of Z-DNA', 'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH',
                     'Breslauer_dS', 'Electron_interaction', 'Hartman_trans_free_energy', 'Helix-Coil_transition',
                     'Ivanov_BA_transition', 'Lisser_BZ_transition', 'Polar_interaction', 'SantaLucia_dG',
                     'SantaLucia_dH', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Stacking_energy',
                     'Sugimoto_dG', 'Sugimoto_dH', 'Sugimoto_dS', 'Watson-Crick_interaction', 'Twist', 'Tilt', 'Roll',
                     'Shift', 'Slide', 'Rise']
    triphyche_list = ['Dnase I', 'Bendability (DNAse)', 'Bendability (consensus)', 'Trinucleotide GC Content',
                      'Nucleosome positioning', 'Consensus_roll', 'Consensus-Rigid', 'Dnase I-Rigid', 'MW-Daltons',
                      'MW-kg', 'Nucleosome', 'Nucleosome-Rigid']

    # Set and check physicochemical properties.
    phyche_list = []
    if k == 2:
        phyche_list = diphyche_list
    elif k == 3:
        phyche_list = triphyche_list

    try:
        if all_property is True:
            phyche_index = phyche_list
        else:
            for e in phyche_index:
                if e not in phyche_list:
                    error_info = 'Sorry, the physicochemical properties ' + e + ' is not exit.'
                    raise NameError(error_info)
    except NameError:
        raise

    # Generate phyche_value and sequence_list.


    phyche_value = extend_phyche_index(get_phyche_index(k, phyche_index), extra_phyche_index)
    sequence_list = get_data(input_data)

    return sequence_list, phyche_value


class PseDNC():
    def __init__(self, lamada=3, w=0.05):
        self.lamada = lamada
        self.w = w
        self.k = 2
        check_psenac(self.lamada, self.w, self.k)

    def make_vec(self, input_data, extra_phyche_index=None):
        """Make PseDNC vector.

        :param input_data: file type or handle.
        :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                         the value is its physicochemical property value (list).
                                   It means the user-defined physicochemical indices.
        """
        sequence_list, phyche_value = get_sequence_list_and_phyche_value_psednc(input_data, extra_phyche_index)


        vector = make_pseknc_vector(sequence_list, self.lamada, self.w, self.k, phyche_value, theta_type=1)

        return vector


class PseKNC():
    """This class should be used to make PseKNC vector."""

    def __init__(self, k=3, lamada=1, w=0.5):
        """
        :param k: k-tuple.
        """
        self.k = k
        self.lamada = lamada
        self.w = w
        check_psenac(self.lamada, self.w, self.k)

    def make_vec(self, input_data, extra_phyche_index=None):
        """Make PseKNC vector.

        :param input_data: file type or handle.
        :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                         the value is its physicochemical property value (list).
                                   It means the user-defined physicochemical indices.
        """
        sequence_list, phyche_value = get_sequence_list_and_phyche_value_pseknc(input_data, extra_phyche_index)


        return make_old_pseknc_vector(sequence_list, self.lamada, self.w, self.k, phyche_value, theta_type=1)


class PCPseDNC():
    def __init__(self, lamada=1, w=0.05):
        self.lamada = lamada
        self.w = w
        self.k = 2
        check_psenac(self.lamada, self.w, self.k)

    def make_vec(self, input_data, phyche_index=None, all_property=False, extra_phyche_index=None):
        """Make a PCPseDNC vector.

        :param input_data: file object or sequence list.
        :param phyche_index: physicochemical properties list.
        :param all_property: choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                         the value is its physicochemical property value (list).
                                   It means the user-defined physicochemical indices.
        """
        # Make vector.
        sequence_list, phyche_value = get_sequence_list_and_phyche_value(input_data, self.k, phyche_index,
                                                                         extra_phyche_index, all_property)


        vector = make_pseknc_vector(sequence_list, self.lamada, self.w, self.k, phyche_value, theta_type=1)

        return vector


class PCPseTNC():
    def __init__(self, lamada=1, w=0.05):
        self.lamada = lamada
        self.w = w
        self.k = 3
        check_psenac(self.lamada, self.w, self.k)

    def make_vec(self, input_data, phyche_index=None, all_property=False, extra_phyche_index=None):
        """Make a PCPseDNC vector.

        :param input_data: file object or sequence list.
        :param phyche_index: physicochemical properties list.
        :param all_property: choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                         the value is its physicochemical property value (list).
                                   It means the user-defined physicochemical indices.
        """
        sequence_list, phyche_value = get_sequence_list_and_phyche_value(input_data, self.k, phyche_index,
                                                                         extra_phyche_index, all_property)
        # Make vector.


        vector = make_pseknc_vector(sequence_list, self.lamada, self.w, self.k, phyche_value, theta_type=1)

        return vector


class SCPseDNC():
    def __init__(self, lamada=1, w=0.05):
        self.lamada = lamada
        self.w = w
        self.k = 2
        check_psenac(self.lamada, self.w, self.k)

    def make_vec(self, input_data, phyche_index=None, all_property=False, extra_phyche_index=None):
        """Make a SCPseDNC vector.

        :param input_data: file object or sequence list.
        :param phyche_index: physicochemical properties list.
        :param all_property: choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                         the value is its physicochemical property value (list).
                                   It means the user-defined physicochemical indices.
        """
        sequence_list, phyche_value = get_sequence_list_and_phyche_value(input_data, self.k, phyche_index,
                                                                         extra_phyche_index, all_property)
        # Make vector.


        vector = make_pseknc_vector(sequence_list, self.lamada, self.w, self.k, phyche_value, theta_type=2)

        return vector


class SCPseTNC():
    def __init__(self, lamada=1, w=0.05):
        self.lamada = lamada
        self.w = w
        self.k = 3
        check_psenac(self.lamada, self.w, self.k)

    def make_vec(self, input_data, phyche_index=None, all_property=False, extra_phyche_index=None):
        """Make a SCPseTNC vector.

        :param input_data: file object or sequence list.
        :param phyche_index: physicochemical properties list.
        :param all_property: choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                         the value is its physicochemical property value (list).
                                   It means the user-defined physicochemical indices.
        """
        sequence_list, phyche_value = get_sequence_list_and_phyche_value(input_data, self.k, phyche_index,
                                                                         extra_phyche_index, all_property)
        # Make vector.


        vector = make_pseknc_vector(sequence_list, self.lamada, self.w, self.k, phyche_value, theta_type=2)

        return vector




ALPHABET = 'ACGT'


"""Used for process original data."""


def make_kmer_list(k, alphabet):
    try:
        return ["".join(e) for e in itertools.product(alphabet, repeat=k)]
    except TypeError:
        print("TypeError: k must be an inter and larger than 0, alphabet must be a string.")
        raise TypeError
    except ValueError:
        print("TypeError: k must be an inter and larger than 0")
        raise ValueError

        
        
def extend_phyche_index(original_index, extend_index):
    """Extend {phyche:[value, ... ]}"""
    if extend_index is None or len(extend_index) == 0:
        return original_index
    for key in list(original_index.keys()):
        original_index[key].extend(extend_index[key])
    return original_index


def get_phyche_factor_dic(k):
    """Get all {nucleotide: [(phyche, value), ...]} dict."""
    full_path = os.path.realpath(__file__)
    if 2 == k:
        file_path = "%s/data/mmc3.data" % os.path.dirname(full_path)
    elif 3 == k:
        file_path = "%s/data/mmc4.data" % os.path.dirname(full_path)
    else:
        sys.stderr.write("The k can just be 2 or 3.")
        sys.exit(0)

    try:
        with open(file_path, 'rb') as f:
            phyche_factor_dic = pickle.load(f)
    except:
        with open(file_path, 'r') as f:
            phyche_factor_dic = pickle.load(f)

    return phyche_factor_dic


def get_phyche_index(k, phyche_list):
    """get phyche_value according phyche_list."""
    phyche_value = {}
    if 0 == len(phyche_list):
        for nucleotide in make_kmer_list(k, ALPHABET):
            phyche_value[nucleotide] = []
        return phyche_value

    nucleotide_phyche_value = get_phyche_factor_dic(k)
    for nucleotide in make_kmer_list(k, ALPHABET):
        if nucleotide not in phyche_value:
            phyche_value[nucleotide] = []
        for e in nucleotide_phyche_value[nucleotide]:
            if e[0] in phyche_list:
                phyche_value[nucleotide].append(e[1])

    return phyche_value

class Seq:
    def __init__(self, name, seq, no):
        self.name = name
        self.seq = seq.upper()
        self.no = no
        self.length = len(seq)

    def __str__(self):
        """Output seq when 'print' method is called."""
        return "%s\tNo:%s\tlength:%s\n%s" % (self.name, str(self.no), str(self.length), self.seq)


def is_under_alphabet(s, alphabet):
    """Judge the string is within the scope of the alphabet or not.

    :param s: The string.
    :param alphabet: alphabet.

    Return True or the error character.
    """
    for e in s:
        if e not in alphabet:
            return e

    return True


def is_fasta(seq):
    """Judge the Seq object is in FASTA format.
    Two situation:
    1. No seq name.
    2. Seq name is illegal.
    3. No sequence.

    :param seq: Seq object.
    """
    if not seq.name:
        raise ValueError(" ".join(["Error, sequence", str(seq.no), "has no sequence name."]))
    if -1 != seq.name.find('>'):
        raise ValueError(" ".join(["Error, sequence", str(seq.no), "name has > character."]))
    if 0 == seq.length:
        raise ValueError(" ".join(["Error, sequence", str(seq.no), "is null."]))

    return True


def read_fasta(f):
    """Read a fasta file.

    :param f: HANDLE to input. e.g. sys.stdin, or open(<file>)

    Return Seq obj list.
    """
    name, seq = '', ''
    count = 0
    seq_list = []
    lines = f.readlines()
    for line in lines:
        if not line:
            break

        if '>' == line[0]:
            if 0 != count or (0 == count and seq != ''):
                if is_fasta(Seq(name, seq, count)):
                    seq_list.append(Seq(name, seq, count))

            seq = ''
            name = line[1:].strip()
            count += 1
        else:
            seq += line.strip()

    count += 1
    if is_fasta(Seq(name, seq, count)):
        seq_list.append(Seq(name, seq, count))

    return seq_list


def read_fasta_yield(f):
    """Yields a Seq object.

    :param f: HANDLE to input. e.g. sys.stdin, or open(<file>)
    """
    name, seq = '', ''
    count = 0
    while True:
        line = f.readline()
        if not line:
            break

        if '>' == line[0]:
            if 0 != count or (0 == count and seq != ''):
                if is_fasta(Seq(name, seq, count)):
                    yield Seq(name, seq, count)

            seq = ''
            name = line[1:].strip()
            count += 1
        else:
            seq += line.strip()

    if is_fasta(Seq(name, seq, count)):
        yield Seq(name, seq, count)


def read_fasta_check_dna(f):
    """Read the fasta file, and check its legality.

    :param f: HANDLE to input. e.g. sys.stdin, or open(<file>)

    Return the seq list.
    """
    seq_list = []
    for e in read_fasta_yield(f):
        res = is_under_alphabet(e.seq, ALPHABET)
        if res:
            seq_list.append(e)
        else:
            raise ValueError(" ".join(["Sorry, sequence", str(e.no), "has character", str(res),
                                       "(The character must be A or C or G or T)"]))

    return seq_list


def get_sequence_check_dna(f):
    """Read the fasta file.

    Input: f: HANDLE to input. e.g. sys.stdin, or open(<file>)

    Return the sequence list.
    """
    sequence_list = []
    for e in read_fasta_yield(f):
        # print e
        res = is_under_alphabet(e.seq, ALPHABET)
        if res is not True:
            raise ValueError(" ".join(["Sorry, sequence", str(e.no), "has character", str(res),
                                       "(The character must be A, C, G or T)"]))
        else:
            sequence_list.append(e.seq)

    return sequence_list


def is_sequence_list(sequence_list):
    """Judge the sequence list is within the scope of alphabet and change the lowercase to capital."""
    count = 0
    new_sequence_list = []

    for e in sequence_list:
        e = e.upper()
        count += 1
        res = is_under_alphabet(e, ALPHABET)
        if res is not True:
            raise ValueError(" ".join(["Sorry, sequence", str(count), "has illegal character", str(res),
                                       "(The character must be A, C, G or T)"]))
        else:
            new_sequence_list.append(e)

    return new_sequence_list


def get_data(input_data, desc=False):
    """Get sequence data from file or list with check.

    :param input_data: type file or list
    :param desc: with this option, the return value will be a Seq object list(it only works in file object).
    :return: sequence data or shutdown.
    """
    if hasattr(input_data, 'read'):
        if desc is False:
            return get_sequence_check_dna(input_data)
        else:
            return read_fasta_check_dna(input_data)
    elif isinstance(input_data, list):
        input_data = is_sequence_list(input_data)
        if input_data is not False:
            return input_data
    else:
        raise ValueError("Sorry, the parameter in get_data method must be list or file type.")


"""Some basic function for generate feature vector."""


def frequency(tol_str, tar_str):
    """Generate the frequency of tar_str in tol_str.

    :param tol_str: mother string.
    :param tar_str: substring.
    """
    i, j, tar_count = 0, 0, 0
    len_tol_str = len(tol_str)
    len_tar_str = len(tar_str)
    while i < len_tol_str and j < len_tar_str:
        if tol_str[i] == tar_str[j]:
            i += 1
            j += 1
            if j >= len_tar_str:
                tar_count += 1
                i = i - j + 1
                j = 0
        else:
            i = i - j + 1
            j = 0

    return tar_count


def write_libsvm(vector_list, label_list, write_file):
    """Write the vector into disk in livSVM format."""
    len_vector_list = len(vector_list)
    len_label_list = len(label_list)
    if len_vector_list == 0:
        raise ValueError("The vector is none.")
    if len_label_list == 0:
        raise ValueError("The label is none.")
    if len_vector_list != len_label_list:
        raise ValueError("The length of vector and label is different.")

    with open(write_file, 'w') as f:
        len_vector = len(vector_list[0])
        for i in range(len_vector_list):
            temp_write = str(label_list[i])
            for j in range(0, len_vector):
                temp_write += ' ' + str(j + 1) + ':' + str(vector_list[i][j])
            f.write(temp_write)
            f.write('\n')


def generate_phyche_value(k, phyche_index=None, all_property=False, extra_phyche_index=None):
    """Combine the user selected phyche_list, is_all_property and extra_phyche_index to a new standard phyche_value."""
    if phyche_index is None:
        phyche_index = []
    if extra_phyche_index is None:
        extra_phyche_index = {}

    diphyche_list = ['Base stacking', 'Protein induced deformability', 'B-DNA twist', 'Dinucleotide GC Content',
                     'A-philicity', 'Propeller twist', 'Duplex stability:(freeenergy)',
                     'Duplex tability(disruptenergy)', 'DNA denaturation', 'Bending stiffness', 'Protein DNA twist',
                     'Stabilising energy of Z-DNA', 'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH',
                     'Breslauer_dS', 'Electron_interaction', 'Hartman_trans_free_energy', 'Helix-Coil_transition',
                     'Ivanov_BA_transition', 'Lisser_BZ_transition', 'Polar_interaction', 'SantaLucia_dG',
                     'SantaLucia_dH', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Stacking_energy',
                     'Sugimoto_dG', 'Sugimoto_dH', 'Sugimoto_dS', 'Watson-Crick_interaction', 'Twist', 'Tilt',
                     'Roll', 'Shift', 'Slide', 'Rise']
    triphyche_list = ['Dnase I', 'Bendability (DNAse)', 'Bendability (consensus)', 'Trinucleotide GC Content',
                      'Nucleosome positioning', 'Consensus_roll', 'Consensus-Rigid', 'Dnase I-Rigid', 'MW-Daltons',
                      'MW-kg', 'Nucleosome', 'Nucleosome-Rigid']

    # Set and check physicochemical properties.
    if 2 == k:
        if all_property is True:
            phyche_index = diphyche_list
        else:
            for e in phyche_index:
                if e not in diphyche_list:
                    raise ValueError(" ".join(["Sorry, the physicochemical properties", e, "is not exit."]))
    elif 3 == k:
        if all_property is True:
            phyche_index = triphyche_list
        else:
            for e in phyche_index:
                if e not in triphyche_list:
                    raise ValueError(" ".join(["Sorry, the physicochemical properties", e, "is not exit."]))

    # Generate phyche_value.


    return extend_phyche_index(get_phyche_index(k, phyche_index), extra_phyche_index)


def convert_phyche_index_to_dict(phyche_index):
    """Convert phyche index from list to dict."""
    # for e in phyche_index:
    #     print e
    len_index_value = len(phyche_index[0])
    k = 0
    for i in range(1, 10):
        if len_index_value < 4**i:
            raise ValueError("Sorry, the number of each index value is must be 4^k.")
        if len_index_value == 4**i:
            k = i
            break

    kmer_list = make_kmer_list(k, ALPHABET)
    # print kmer_list
    len_kmer = len(kmer_list)
    phyche_index_dict = {}
    for kmer in kmer_list:
        phyche_index_dict[kmer] = []
    # print phyche_index_dict
    phyche_index = list(zip(*phyche_index))
    for i in range(len_kmer):
        phyche_index_dict[kmer_list[i]] = list(phyche_index[i])

    return phyche_index_dict


def standard_deviation(value_list):
    """Return standard deviation."""
    n = len(value_list)
    average_value = sum(value_list) * 1.0 / n
    return sqrt(sum([pow(e - average_value, 2) for e in value_list]) * 1.0 / (n - 1))


def normalize_index(phyche_index, is_convert_dict=False):
    """Normalize the physicochemical index."""
    normalize_phyche_value = []
    for phyche_value in phyche_index:
        average_phyche_value = sum(phyche_value) * 1.0 / len(phyche_value)
        sd_phyche = standard_deviation(phyche_value)
        normalize_phyche_value.append([round((e - average_phyche_value) / sd_phyche, 2) for e in phyche_value])

    if is_convert_dict is True:
        return convert_phyche_index_to_dict(normalize_phyche_value)

    return normalize_phyche_value




ALPHABET = 'ACGT'

def frequency(tol_str, tar_str):
    """Generate the frequency of tar_str in tol_str.

    :param tol_str: mother string.
    :param tar_str: substring.
    """
    i, j, tar_count = 0, 0, 0
    len_tol_str = len(tol_str)
    len_tar_str = len(tar_str)
    while i < len_tol_str and j < len_tar_str:
        if tol_str[i] == tar_str[j]:
            i += 1
            j += 1
            if j >= len_tar_str:
                tar_count += 1
                i = i - j + 1
                j = 0
        else:
            i = i - j + 1
            j = 0

    return tar_count


def make_kmer_list(k, alphabet):
    try:
        return ["".join(e) for e in itertools.product(alphabet, repeat=k)]
    except TypeError:
        print("TypeError: k must be an inter and larger than 0, alphabet must be a string.")
        raise TypeError
    except ValueError:
        print("TypeError: k must be an inter and larger than 0")
        raise ValueError

        
def extend_phyche_index(original_index, extend_index):
    """Extend {phyche:[value, ... ]}"""
    if extend_index is None or len(extend_index) == 0:
        return original_index
    for key in list(original_index.keys()):
        original_index[key].extend(extend_index[key])
    return original_index


def get_phyche_factor_dic(k):
    """Get all {nucleotide: [(phyche, value), ...]} dict."""
    full_path = os.path.realpath(__file__)
    if 2 == k:
        file_path = "%s/data/mmc3.data" % os.path.dirname(full_path)
    elif 3 == k:
        file_path = "%s/data/mmc4.data" % os.path.dirname(full_path)
    else:
        sys.stderr.write("The k can just be 2 or 3.")
        sys.exit(0)

    try:
        with open(file_path, 'rb') as f:
            phyche_factor_dic = pickle.load(f)
    except:
        with open(file_path, 'r') as f:
            phyche_factor_dic = pickle.load(f)

    return phyche_factor_dic


def get_phyche_index(k, phyche_list):
    """get phyche_value according phyche_list."""
    phyche_value = {}
    if 0 == len(phyche_list):
        for nucleotide in make_kmer_list(k, ALPHABET):
            phyche_value[nucleotide] = []
        return phyche_value

    nucleotide_phyche_value = get_phyche_factor_dic(k)
    for nucleotide in make_kmer_list(k, ALPHABET):
        if nucleotide not in phyche_value:
            phyche_value[nucleotide] = []
        for e in nucleotide_phyche_value[nucleotide]:
            if e[0] in phyche_list:
                phyche_value[nucleotide].append(e[1])

    return phyche_value


def parallel_cor_function(nucleotide1, nucleotide2, phyche_index):
    """Get the cFactor.(Type1)"""
    temp_sum = 0.0
    phyche_index_values = list(phyche_index.values())
    len_phyche_index = len(phyche_index_values[0])
    for u in range(len_phyche_index):
        temp_sum += pow(float(phyche_index[nucleotide1][u]) - float(phyche_index[nucleotide2][u]), 2)

    return temp_sum / len_phyche_index


def series_cor_function(nucleotide1, nucleotide2, big_lamada, phyche_value):
    """Get the series correlation Factor(Type 2)."""
    return float(phyche_value[nucleotide1][big_lamada]) * float(phyche_value[nucleotide2][big_lamada])


def get_parallel_factor(k, lamada, sequence, phyche_value):
    """Get the corresponding factor theta list."""
    theta = []
    l = len(sequence)

    for i in range(1, lamada + 1):
        temp_sum = 0.0
        for j in range(0, l - k - i + 1):
            nucleotide1 = sequence[j: j+k]
            nucleotide2 = sequence[j+i: j+i+k]
            temp_sum += parallel_cor_function(nucleotide1, nucleotide2, phyche_value)

        theta.append(temp_sum / (l - k - i + 1))

    return theta


def get_series_factor(k, lamada, sequence, phyche_value):
    """Get the corresponding series factor theta list."""
    theta = []
    l_seq = len(sequence)
    temp_values = list(phyche_value.values())
    max_big_lamada = len(temp_values[0])

    for small_lamada in range(1, lamada + 1):
        for big_lamada in range(max_big_lamada):
            temp_sum = 0.0
            for i in range(0, l_seq - k - small_lamada + 1):
                nucleotide1 = sequence[i: i+k]
                nucleotide2 = sequence[i+small_lamada: i+small_lamada+k]
                temp_sum += series_cor_function(nucleotide1, nucleotide2, big_lamada, phyche_value)

            theta.append(temp_sum / (l_seq - k - small_lamada + 1))

    return theta


def make_pseknc_vector(sequence_list, lamada, w, k, phyche_value, theta_type=1):
    """Generate the pseknc vector."""
    kmer = make_kmer_list(k, ALPHABET)
    vector = []

    for sequence in sequence_list:
        if len(sequence) < k or lamada + k > len(sequence):
            error_info = "Sorry, the sequence length must be larger than " + str(lamada + k)
            sys.stderr.write(error_info)
            sys.exit(0)

        # Get the nucleotide frequency in the DNA sequence.
        fre_list = [frequency(sequence, str(key)) for key in kmer]
        fre_sum = float(sum(fre_list))

        # Get the normalized occurrence frequency of nucleotide in the DNA sequence.
        fre_list = [e / fre_sum for e in fre_list]

        # Get the theta_list according the Equation 5.
        if 1 == theta_type:
            theta_list = get_parallel_factor(k, lamada, sequence, phyche_value)
        elif 2 == theta_type:
            theta_list = get_series_factor(k, lamada, sequence, phyche_value)
        theta_sum = sum(theta_list)

        # Generate the vector according the Equation 9.
        denominator = 1 + w * theta_sum

        temp_vec = [round(f / denominator, 3) for f in fre_list]
        for theta in theta_list:
            temp_vec.append(round(w * theta / denominator, 4))

        vector.append(temp_vec)

    return vector


def get_parallel_factor_psednc(lamada, sequence, phyche_value):
    """Get the corresponding factor theta list.
       This def is just for dinucleotide."""
    theta = []
    l = len(sequence)

    for i in range(1, lamada + 1):
        temp_sum = 0.0
        for j in range(0, l - 1 - lamada):
            nucleotide1 = sequence[j] + sequence[j + 1]
            nucleotide2 = sequence[j + i] + sequence[j + i + 1]
            temp_sum += parallel_cor_function(nucleotide1, nucleotide2, phyche_value)

        theta.append(temp_sum / (l - i - 1))

    return theta


def make_old_pseknc_vector(sequence_list, lamada, w, k, phyche_value, theta_type=1):
    """Generate the pseknc vector."""
    kmer = make_kmer_list(k, ALPHABET)
    vector = []

    for sequence in sequence_list:
        if len(sequence) < k or lamada + k > len(sequence):
            error_info = "Sorry, the sequence length must be larger than " + str(lamada + k)
            sys.stderr.write(error_info)
            sys.exit(0)

        # Get the nucleotide frequency in the DNA sequence.
        fre_list = [frequency(sequence, str(key)) for key in kmer]
        fre_sum = float(sum(fre_list))

        # Get the normalized occurrence frequency of nucleotide in the DNA sequence.
        fre_list = [e / fre_sum for e in fre_list]

        # Get the theta_list according the Equation 5.
        if 1 == theta_type:
            theta_list = get_parallel_factor_psednc(lamada, sequence, phyche_value)
        elif 2 == theta_type:
            theta_list = get_series_factor(k, lamada, sequence, phyche_value)
        theta_sum = sum(theta_list)

        # Generate the vector according the Equation 9.
        denominator = 1 + w * theta_sum

        temp_vec = [f / denominator for f in fre_list]
        for theta in theta_list:
            temp_vec.append(w * theta / denominator)

        vector.append(temp_vec)

    return vector
def get_feature_pseKnc(seq,k):
    phyche_index = [[1.019, -0.918, 0.488, 0.567, 0.567, -0.070, -0.579, 0.488, -0.654, -2.455, -0.070, -0.918, 1.603,
                     -0.654, 0.567, 1.019]]
    pseknc=PseKNC(k=k)
    vec = pseknc.make_vec([seq],extra_phyche_index=normalize_index(phyche_index, is_convert_dict=True))
    return vec[0]

