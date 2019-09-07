#!/bin/python
'''
'''
from feasibgs import catalogs as Cat

def legacy_1400deg2_test_region(rlimit=21.):
    ''' construct combined sweeps catalog from legacy 
    '''
    leg = Cat.Legacy() 
    #leg._collect_1400deg2_test()
    leg._1400deg2_test(rlimit=rlimit)
    return None


def legacy_1400deg2_test_region_collect_sweeps(rlimit=21.):
    leg = Cat.Legacy() 
    leg._collect_1400deg2_test(rlimit=rlimit)
    return None


if __name__=="__main__":
    #legacy_1400deg2_test_region_collect_sweeps(rlimit=21.)
    legacy_1400deg2_test_region(rlimit=21.)

