import sys, os, time, fnmatch, itertools, re
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from scipy import sparse as sp
from math import log
from collections import defaultdict

def get_tree_df(fname):
    with open(fname, 'r') as f:
        rows = []
        for line in f:
            if line[0] == "#":
                continue
            
            line = line.strip().split(' ')
            pid = int(line[2].strip('"'))
            cl = line[0]
            rank = float(line[1])
            
            rows.append( (pid, rank, cl) )
    df = pd.DataFrame(rows, columns=['pid', 'paper_rank', 'cl'])
    df['cl_bottom'] = df.cl.apply(lambda x: ':'.join(x.split(':')[:-1])).astype('category')
#     df['cl_top'] = df.cl.apply(lambda x: x.split(':')[0]).astype('category')
    
    df = df.sort_values('pid')

    return df


def _get_cl_level(x, level=2):
    spl = x.split(':')[:-1]
    if level == -1:
        # special case for level == -1
        return ':'.join(spl)
    if len(spl) < level:
        return 'NA'
    else:
        return ':'.join(spl[:level])


def get_contingency_matrix(left, right):
    # left and right are arrays of cluster assignments (the 'codes' for pandas categories)
    return contingency_matrix(left, right, sparse=True)

def get_assignments(df, level):
    x = df.cl.apply(_get_cl_level, level=level)
    x = x.astype('category')
    return x

def get_category_name_from_code(s, code):
    return s.cat.categories[code]

def all_pmis_from_contingency(contingency, normalized=False):
    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    contingency_nzx, contingency_nzy, contingency_nz_val = sp.find(contingency)
    pi_diag = sp.diags(pi)
    pj_diag = sp.diags(pj)
    m = np.dot(pi_diag, contingency)
    m = np.dot(m, pj_diag)
    m = m.power(-1)
    m = m.multiply(contingency)
    m = m * contingency_sum
    mnzx, mnzy, mnz_val = sp.find(m)
    mnz_val = np.log(mnz_val)
    if normalized:
        mnz_val = mnz_val / (-np.log(contingency_nz_val/float(contingency_sum)))
    return np.array(zip(mnzx, mnzy, mnz_val), dtype=[('x_idx', int), ('y_idx', int), ('pmi', np.float)])


def get_all_pmis_one_level(df_left, df_right, level_left=-1, level_right=-1, normalized=True):
    # get the pmis for a (bottom level) cluster on the left against the clusters of <level> on the right
    
    left = get_assignments(df_left, level=level_left)
    left_codes = left.cat.codes
    right = get_assignments(df_right, level=level_right)
    right_codes = right.cat.codes
    contingency = get_contingency_matrix(left_codes, right_codes)
    
    pmis = all_pmis_from_contingency(contingency, normalized=normalized)
    return pmis, left, right

def find_max_depth(df_list):
    max_depth = 0
    for df in df_list:
        depth = df.cl_bottom.apply(lambda x: len(x.split(':')))
        this_max_depth = depth.max()
        if this_max_depth > max_depth:
            max_depth = this_max_depth
    return max_depth

def align_two_dfs(df1, df2, colname='pid'):
    df1 = df1[df1[colname].isin(df2[colname])]
    df2 = df2[df2[colname].isin(df1[colname])]
    return df1, df2


def get_all_levels_data(left_df, right_df, max_depth=None):
    # may take some time
    
    if max_depth is None:
        max_depth = find_max_depth([left_df, right_df])
    
    # make both dfs contain the same papers
    left_df, right_df = align_two_dfs(left_df, right_df)
    
    levels = range(1,max_depth)
    data = defaultdict(dict)
    for level_left, level_right in itertools.product(levels, repeat=2):
        this_data = get_all_pmis_one_level(left_df, right_df, level_left, level_right)
        data[level_left][level_right] = this_data
    return data

def data_to_df(data):
    dfs_pmi = []
    for level_left in data:
        for level_right in data[level_left]:
            this_data = data[level_left][level_right]
            this_df_pmi = pd.DataFrame(this_data[0])
            this_df_pmi['level_left'] = level_left
            this_df_pmi['level_right'] = level_right
#             this_df_pmi['cl_left'] = this_df_pmi['x_idx'].apply(lambda x: get_category_name_from_code(this_data[1], x))
#             this_df_pmi['cl_right'] = this_df_pmi['y_idx'].apply(lambda x: get_category_name_from_code(this_data[2], x))
            this_df_pmi['cl_left'] = this_df_pmi['x_idx'].map(pd.Series(this_data[1].cat.categories))
            this_df_pmi['cl_right'] = this_df_pmi['y_idx'].map(pd.Series(this_data[2].cat.categories))
            dfs_pmi.append(this_df_pmi)
    return pd.concat(dfs_pmi)

def get_all_levels_df(left_df, right_df, max_depth=None, return_data=True):
    data = get_all_levels_data(left_df, right_df, max_depth=max_depth)
    df = data_to_df(data)
    if return_data is True:
        return df, data
    else:
        return df

# def _get_cl_name(row, data, side='right'):
#     level_left = row['level_left']
#     level_right = row['level_right']
#     if side == 'right':
#         idx = int(row['y_idx'])
#         side_idx = 2
#     elif side == 'left':
#         idx = int(row['x_idx'])
#         side_idx = 1
#     return get_category_name_from_code(data[level_left][level_right][side_idx], idx)


import logging
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    logger.debug("Parsing 1st input file: {}".format(args.infile_left))
    df_left = get_tree_df(args.infile_left)
    logger.debug("Parsing 2nd input file: {}".format(args.infile_right))
    df_right = get_tree_df(args.infile.right)
    logger.debug("Getting data")
    df_data = get_all_levels_df(df_left, df_right, return_data=False)
    logger.debug("Writing to file: {}".format(args.outfile))
    df_data.to_csv(args.outfile, index=False)

if __name__ == "__main__":
    total_start = time.time()
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description="Given two clusterings (on same/similar data) compare clusters using PMI (Pointwise Mutual Information)")
    parser.add_argument("infile_left", help="filename for the 1st input tree file containing a clustering")
    parser.add_argument("infile_right", help="filename for the 2nd input tree file containing a clustering")
    parser.add_argument("outfile", help="output filename (CSV)")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    main(args)
    total_end = time.time()
    logger.info('all finished. total time: {:.2f} seconds'.format(total_end-total_start))


