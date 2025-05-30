import pandas as pd
import pickle
import sys
import itertools
from collections import Counter
import math

from tqdm import tqdm

codon_to_aa = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'', 'TAG':'',
    'TGC':'C', 'TGT':'C', 'TGA':'', 'TGG':'W'
    }


ALPHABET = 'ACGT'
def tf(word, count):
    return count[word] / sum(count.values())


def idf(word, count_list):
    n_contain = sum([1 for count in count_list if word in count])
    return math.log(len(count_list) / (1 + n_contain))


def tf_idf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)
def readspots(seq):
    res=[]
    for i in seq:
        spot=""
        spot = spot + i.replace("\n", '').replace("\t",'')
        res.append(spot)
    return res
def readspots_file(path):
    with open(path,"r") as f:
        text=f.readlines()
    ans=[]
    current=[]
    spot=""
    for txt in text:
        if ">" in txt:
            if len(current)!=0:
                current.append(spot)
                ans.append(current)
            current=[]
            spot=""
            # 添加名字
            current.append(txt.replace("\n",''))
        else:
            spot=spot+txt.replace("\n",'')
    current.append(spot)
    ans.append(current)
    return ans
def get_k_mer(k_ver,words):
    ans=[]
    for k in k_ver:
        ans.extend(["".join(e) for e in itertools.product(words, repeat=k)])
    return ans
def k_mer(k,fasta):
    l=0
    r=k+l
    mers=[]
    while r<len(fasta)+1:
        mers.append(fasta[l:r])
        l+=1
        r=k+l
    return mers
def get_word_list(kmer_list):
    cold_path = "data/coldspots.fasta"
    hot_path = "data/hotspots.fasta"
    cold_fasta = readspots_file(cold_path)
    hot_fasta = readspots_file(hot_path)
    names = []
    fastas = []
    labels = []
    words_list = []
    for name, fasta in enumerate(hot_fasta):
        names.append(fasta[0])
        fastas.append(fasta[1])
        labels.append(1)
    for name, fasta in enumerate(cold_fasta):
        names.append(fasta[0])
        fastas.append(fasta[1])
        labels.append(0)
    # 清理序列
    for i in range(len(fastas)):
        fastas[i] = fastas[i].replace("/n", "")
        fastas[i] = fastas[i].replace("/r", "")
        fastas[i] = fastas[i].replace(" ", "")
    data = pd.DataFrame(zip(names, fastas, labels), columns=["name", "fasta", "label"])
    for fasta in fastas:
        current_word = []
        for kmer in kmer_list:
            num = fasta.count(kmer)
            for i in range(num):
                current_word.append(kmer)
        words_list.append(current_word)
    return words_list
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
def extend_phyche_index(original_index, extend_index):
    """Extend {phyche:[value, ... ]}"""
    if extend_index is None or len(extend_index) == 0:
        return original_index
    for key in list(original_index.keys()):
        original_index[key].extend(extend_index[key])
    return original_index
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

        theta_list = get_parallel_factor_psednc(lamada, sequence, phyche_value)

        theta_sum = sum(theta_list)

        # Generate the vector according the Equation 9.
        denominator = 1 + w * theta_sum

        temp_vec = [round(f / denominator, 3) for f in fre_list]
        for theta in theta_list:
            temp_vec.append(round(w * theta / denominator, 4))

        vector.append(temp_vec)

    return vector
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
        theta_list = get_parallel_factor(k, lamada, sequence, phyche_value)
        theta_sum = sum(theta_list)

        # Generate the vector according the Equation 9.
        denominator = 1 + w * theta_sum

        temp_vec = [round(f / denominator, 3) for f in fre_list]
        for theta in theta_list:
            temp_vec.append(round(w * theta / denominator, 4))

        vector.append(temp_vec)

    return vector
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
def parallel_cor_function(nucleotide1, nucleotide2, phyche_index):
    """Get the cFactor.(Type1)"""
    temp_sum = 0.0
    phyche_index_values = list(phyche_index.values())
    len_phyche_index = len(phyche_index_values[0])
    for u in range(len_phyche_index):
        temp_sum += pow(float(phyche_index[nucleotide1][u]) - float(phyche_index[nucleotide2][u]), 2)

    return temp_sum / len_phyche_index
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
def standard_deviation(value_list):
    """Return standard deviation."""
    from math import sqrt
    from math import pow
    n = len(value_list)
    average_value = sum(value_list) * 1.0 / n
    return sqrt(sum([pow(e - average_value, 2) for e in value_list]) * 1.0 / (n - 1))
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
class PseDNC():
    def __init__(self, lamada=3, w=0.05):
        self.lamada = lamada
        self.w = w
        self.k = 2
        check_psenac(self.lamada, self.w, self.k)

    def make_psednc_vec(self, input_data, extra_phyche_index=None):
        """Make PseDNC vector.

        :param input_data: file type or handle.
        :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                         the value is its physicochemical property value (list).
                                   It means the user-defined physicochemical indices.
        """
        sequence_list, phyche_value = get_sequence_list_and_phyche_value_psednc(input_data, extra_phyche_index)

        vector = make_pseknc_vector(sequence_list, self.lamada, self.w, self.k, phyche_value, theta_type=1)

        return vector
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

    def make_pseknc_vec(self, input_data, extra_phyche_index=None):
        """Make PseKNC vector.

        :param input_data: file type or handle.
        :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                         the value is its physicochemical property value (list).
                                   It means the user-defined physicochemical indices.
        """
        sequence_list, phyche_value = get_sequence_list_and_phyche_value_pseknc(input_data, extra_phyche_index)

        return make_old_pseknc_vector(sequence_list, self.lamada, self.w, self.k, phyche_value, theta_type=1)
def make_kmer_list(k, alphabet):
    try:
        return ["".join(e) for e in itertools.product(alphabet, repeat=k)]
    except TypeError:
        print("TypeError: k must be an inter and larger than 0, alphabet must be a string.")
        raise TypeError
    except ValueError:
        print("TypeError: k must be an inter and larger than 0")
        raise ValueError
def getGC(seq):
    G_C_num = 0
    for s in seq:
        if s == "G" or s == "C":
            G_C_num += 1
    return float(G_C_num)/len(seq)
def getATGC(seq):
    AT_num = 0
    GC_num = 0
    for s in seq:
        if s == "A" or s == "T":
            AT_num += 1
        if s == "G" or s == "C":
            GC_num += 1
    return float(AT_num) / GC_num
def getZcurver(seq):
    A_num = 0
    G_num = 0
    C_num = 0
    T_num = 0
    for s in seq:
        if s == "A":
            A_num += 1
        elif s == "G":
            G_num += 1
        elif s == "C":
            C_num += 1
        elif s == "T":
            T_num += 1
    return [A_num + G_num - C_num - T_num,A_num + C_num - G_num - T_num,A_num + T_num - G_num - C_num]
def getskew(seq):
    A_num = 0
    G_num = 0
    C_num = 0
    T_num = 0
    for s in seq:
        if s == "A":
            A_num += 1
        elif s == "G":
            G_num += 1
        elif s == "C":
            C_num += 1
        elif s == "T":
            T_num += 1
    return [float(G_num - C_num) / float(G_num + C_num),float(A_num - T_num) / float(A_num + T_num)]
def getkmer(seq):
    words="AGCT"
    k_ver = [1, 2, 3, 4, 5]
    kmer_list = get_k_mer(k_ver, words)
    fasta_kmer = []
    for kmer in kmer_list:
        fasta_kmer.append(seq.count(kmer))
    return fasta_kmer
def gettfidfkmer(seq):
    words = "AGCT"
    k_ver = [3, 4, 5]
    kmer_list = get_k_mer(k_ver, words)
    current_word = []
    for kmer in kmer_list:
        num = seq.count(kmer)
        for i in range(num):
            current_word.append(kmer)
    count=Counter(current_word)
    current_dict = {}
    current_list = []
    with open('count_list.pkl', 'rb') as file:
        count_list = pickle.load(file)
    scores = {word: tf_idf(word, count, count_list) for word in count}
    for word, score in scores.items():
        current_dict[word] = score
    for kmer in kmer_list:
        if kmer in current_dict.keys():
            current_list.append(current_dict[kmer])
        else:
            current_list.append(0) # 先填充0进行尝试
    return current_list
def getpseknc(seq,k=5,l=2):
    pseknc = PseKNC(k=k, lamada=l, w=0.05)
    vec = pseknc.make_pseknc_vec([seq])
    return vec[0]

def getfeatures(seq):
    GC=getGC(seq) # GC含量
    ATGC=getATGC(seq) #AT/GC比例
    Zcurver=getZcurver(seq) # Z-curver曲线
    GCATskew=getskew(seq) # 累计偏斜
    kmer=getkmer(seq) # kmer
    # tfidfkmer=gettfidfkmer(seq) # tf-idf-kmer
    pseknc=getpseknc(seq,5,2) # pseknc
    psednc=getpseknc(seq,2,2) #psednc
    feature=[]
    feature.append(GC)
    feature.append(ATGC)
    feature=feature+Zcurver
    feature=feature+GCATskew
    feature=feature+kmer
    # feature=feature+tfidfkmer
    feature=feature+pseknc
    feature=feature+psednc
    res=[]
    ## 特征选择
    # select_f=[1,20,92,104,114,120,121,123,133,134,156,160,161,167,169,174,177,179,185,192,196,198,203,209,217,222,232,244,245,246,250,258,269,280,284,289,291,297,308,332,334,352,367,368,380,392,402,405,406,413,414,415,416,426,432,435,438,439,442,445,454,456,462,472,481,484,486,492,495,496,497,501,505,516,520,523,524,526,527,530,537,549,551,554,556,567,575,576,578,579,582,585,595,596,601,625,629,633,635,641,645,651,652,657,659,665,669,670,673,678,683,690,695,698,701,721,723,727,734,736,757,762,768,775,783,787,790,796,803,805,806,807,808,811,816,819,824,828,836,840,841,848,853,854,855,856,860,861,862,865,871,873,875,878,900,908,909,911,913,919,922,929,930,935,944,946,950,952,954,957,972,973,981,983,986,987,989,995,996,998,1003,1004,1025,1030,1035,1036,1038,1047,1049,1052,1066,1068,1070,1079,1083,1084,1089,1090,1095,1096,1097,1104,1105,1107,1112,1115,1123,1125,1127,1129,1131,1133,1136,1142,1146,1163,1167,1171,1173,1177,1181,1193,1194,1195,1199,1200,1201,1205,1206,1207,1216,1219,1223,1226,1230,1236,1238,1240,1252,1253,1254,1261,1264,1268,1270,1277,1278,1281,1285,1287,1292,1301,1315,1318,1323,1324,1328,1335,1340,1354,1355,1356,1360,1361,1370]
    # for i in select_f:
    #     res.append(feature[i])
    return feature
def readspots(path):
    with open(path,"r") as f:
        text=f.readlines()
    ans=[]
    current=[]
    spot=""
    for txt in text:
        if ">" in txt:
            if len(current)!=0:
                current.append(spot)
                ans.append(current)
            current=[]
            spot=""
            # 添加名字
            current.append(txt.replace("\n",''))
        else:
            spot=spot+txt.replace("\n",'')
    current.append(spot)
    ans.append(current)
    return ans
def getseqfeatutes(seq,path):
    table_features=[]
    for i in tqdm(range(len(seq))):
        table_features.append(getfeatures(seq[i]))
    data=pd.DataFrame(table_features)
    data.to_csv(path,index=False)