#!/usr/bin/env python

import sys
import numpy as np
import subprocess as sub

pdbfile = sys.argv[1:]

def d3(x1,y1,z1,x2,y2,z2): # 3 dimensional distance formula
    return (((x2 - x1) ** 2) + ((y2 - y1) ** 2) + ((z2 - z1) ** 2)) ** 0.5

def mod_avg(l):
    if len(l) == 0:
        return 0
    else:
        return (sum(l) / len(l))

class Protein:

    def __init__(self, pdb):
        self.pdb = pdb 
        self.atoms = self.readatoms()
        self.calist = self.determine_ca()
        self.cofm = self.cofm2()

    def readatoms(self): # returns list of dictionaries
        atom_list = []
        for line in open(self.pdb):
            if line.split(' ')[0] == 'ATOM':
                new = line.split(' ')
                new = [i for i in new if i]
                try:
                    int(new[5])
                    del new[4]
                except ValueError:
                    pass
                #print(new)
                atom_info = {'Index' : int(new[1]), # creates dictionary for each atom
                             'Element' : new[2],
                             'Residue' : new[3],
                             'Residue Index' : int(new[4]),
                             'X' : float(new[5]),
                             'Y' : float(new[6]),
                             'Z' : float(new[7])}
                atom_list.append(atom_info) 
        return atom_list

    def determine_ca(self): # returns list of indicies of ca atoms in 'self.atoms'
        ca_list = []
        for n in range(len(self.atoms)):
            if self.atoms[n]['Element'] == 'CA':
                ca_list.append(n)
        return ca_list

    ## CA-COFM DIST METHODS ##
    
    def cofm2(self): # finds the center using only CA atoms
        x = []
        y = []
        z = []
        for i in range(len(self.calist)):
            x.append(self.atoms[self.calist[i]]['X'])
            y.append(self.atoms[self.calist[i]]['Y'])
            z.append(self.atoms[self.calist[i]]['Z']) 
        cofm = {'X': float(np.mean(x)),
                'Y': float(np.mean(y)),
                'Z': float(np.mean(z))}
        return cofm

    def all_ca_cofm_dist(self,x=0):
        if x == 0:
            for i in range(len(self.calist)):
                print(d3(self.atoms[self.calist[i]]['X'],self.atoms[self.calist[i]]['Y'],self.atoms[self.calist[i]]['Z'],self.cofm['X'],self.cofm['Y'],self.cofm['Z']), end=',')
            print()
        else:
            for i in range(x):
                try:
                    print(d3(self.atoms[self.calist[i]]['X'],self.atoms[self.calist[i]]['Y'],self.atoms[self.calist[i]]['Z'],self.cofm['X'],self.cofm['Y'],self.cofm['Z']), end=',')
                except IndexError:
                    print(0, end=',')
            print()

def main():

    protein = Protein(pdbfile) # creates 'protein' object
    protein.all_ca_cofm_dist(1000) # runs ca-cofm methods

if __name__ == '__main__':
    main()

