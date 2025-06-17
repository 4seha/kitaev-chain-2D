# %% Import libraries, define functions
import numpy as np
import matplotlib.pyplot as plt

# %% 1D %%
def kitaev_chain(N,mu,t,scp, pbc = False):
    """
    Constructs the Hamiltonian matrix for a Kitaev chain.

    Parameters
    ----------
    N : int
        Number electron sites in chain. 
        Majorana basis correspondence: 2N Majorana modes.
    mu : float
        Chemical potential at each electron site. 
        Majorana basis correspondence: An imaginary on-site coupling between adjacent Majorana modes of the same electron site.
    t : float
        Hopping amplitude between neighboring electron sites. 
        Majorana basis correspondence: Hopping constants between adjacent Majarona modes that belong to different electron sites.
    scp : float
        Superconducting pairing amplitude between neighboring electron sites.
    pbc : bool, optional
        If True, applies periodic boundary conditions (PBC). Default is False.

    Returns

    -------
    H : ndarray
        A 2N x 2N Hermitian Hamiltonian matrix representing the Kitaev chain in the Majorana basis.
    """
    # N: number of electrons
    # t: hopping constant
    # sp: superconducting pairing coefficient
    
    B = 2   # count of Majorana operators (sites)

    M = -1j*mu/2
    Tp = 1j*(scp+t)/2
    Tn = 1j*(scp-t)/2

    offd_vec1 = M * np.ones( B*N - 1)
    offd_vec1[1::2] = Tp
    offd_vec2 = np.zeros(B*N-3,dtype=complex)
    offd_vec2[0::2] = Tn

    H = np.diag(offd_vec1,1) + np.diag(offd_vec2,3)
    
    if pbc == True:
        H[0, -1] = -Tp
        H[B-1, -2] = -Tn
    
    H += H.conj().T

    return H


def energy(H_full, cell_count_in_x = 0, apply_ft_on = None, sites=2):
    """
    Obtains eigenvalues of the given Hamiltonian. If 'apply_ft_on' is
    set as a valid integer, apply Fourier transform beforehand.

    Parameters
    ----------
    H_full : array_like
        Matrix to be Fourier transformed.
    cell_count_in_x : integer
        Amount of unit cells on x-axis (second fastest varying index).
    apply_ft_on : integer
        Axis on which Fourier transform will be applied. If -1, take Fourier
        transform on both axes. If None, returns the eigenvalues without 
        Fourier tranformation. 
    sites : int
        Site count in unit cell
    """
    if apply_ft_on == None:
        Hk = H_full
    else:
        N_x = cell_count_in_x
        if cell_count_in_x == 0:
            N_x = int(H_full.shape[1]/sites)
            # print('ATTENTION: cell_count_in_x is not provided, length divided by count of sites per unit cells as unit cell count.({})'.format(N_x))

        if (apply_ft_on == 0)|(apply_ft_on == 1):
            Hk = ft_hamiltonian(H_full, N_x, axis= apply_ft_on, sites=sites)
        elif (apply_ft_on == -1):
            Hk = ft2_hamiltonian(H_full, N_x, sites=sites)
        else:
            raise Exception('Invalied input for apply_ft_on')
    return np.linalg.eigvals(Hk)

def E_k_kitaev_theory(k_vector,mu,t,scp):
    """
    Returns the theoretical reciprocal space energy band of the Kitaev chain.

    Parameters
    ----------
    k_vector : array_like
        k points of energy band to be evaluated.
    mu : float
        Chemical potential of Dirac fermions.
    t : float
        Hopping constants between of n-neighbor Dirac fermions.
    scp : float
        Superconducting pairing constant of n-neighbor Dirac fermions.
    """
    absE = np.sqrt( (mu + 2*t*np.cos(k_vector))**2 + (2*scp*np.sin(k_vector))**2 )/2
    return  [absE, -absE]

def eigenstates(H_full, cell_count_in_x = 0, apply_ft_on = None, sites=2):
    """
    Obtains eigenvectors of the given Hamiltonian. 
    The 'apply_ft_on' has no effect. Can be implemented later if needed.
    
    Parameters
    ----------
    H_full : array_like
        Matrix to be Fourier transformed.
    cell_count_in_x : integer
        Amount of unit cells on x-axis (second fastest varying index).
    apply_ft_on : integer
        Axis on which Fourier transform will be applied. If -1, take Fourier
        transform on both axes. If None, returns the eigenvalues without 
        Fourier tranformation. 
    sites : int
        Site count in unit cell
    """
    Hk = H_full
    if H_full.shape[1]%sites != 0: raise Exception('Site count (variable \'sites\') don\'t comply with the shape of matrix given as input.')
    if cell_count_in_x == 0: cell_count_in_x = int(H_full.shape[1]/sites)
    eigvals, eigvecs = np.linalg.eig(Hk)
    sorted_eigvecs = eigvecs[:,np.argsort(abs(eigvals), axis=-1)]
    return sorted_eigvecs
# %% 2D %%
def kitaev_plane(N_x,N_y,mu,t_x,scp_x, pbx = 0, pby = 0, t_y=-1,scp_y=-1 ):
    # N: number of electrons
    # t: hopping constant
    # sp: superconducting pairing coefficient
    
    
    if t_y==-1:
        t_y = t_x
    if scp_y == -1:
        scp_y = scp_x

    one = 1     # will use this for changes regarding the Python indexing
    # two = 2     # count of Majorana operators (sites) SORRY IT MAKES CODE HARD TO READ, I WILL GO MAGIC
    
    L = N_x*N_y*2 # Total number of Majoranas
    M = -1j*mu/2
    Xp = 1j*(scp_x+t_x)/2
    Xn = 1j*(scp_x-t_x)/2
    Yp = 1j*(scp_y+t_y)/2
    Yn = 1j*(scp_y-t_y)/2


    H = np.zeros((L,L),dtype=complex) # If copmlex matrix
    for m in range(one, N_x + one):
        #print(N_x)
        for n in range(one, N_y + one):
            H[2*N_x*(n-1) + 2*m     - one,  2*N_x*(n-1) + 2*m - 1 - one] += M
        for n in range(one - pby, N_y - 1 +  one):
            H[2*N_x*n     + 2*m - 1 - one,  2*N_x*(n-1) + 2*m     - one] += Yp
            H[2*N_x*n     + 2*m     - one,  2*N_x*(n-1) + 2*m - 1 - one] += Yn
    for n in range(one, N_y + one):
        for m in range(one, N_x - 1 + one):        
            H[2*N_x*(n-1) + 2*m + 1 - one,  2*N_x*(n-1) + 2*m     - one] += Xp
            H[2*N_x*(n-1) + 2*m + 2 - one,  2*N_x*(n-1) + 2*m - 1 - one] += Xn
    if pbx == 1:
        for n in range(one, N_y + one):
            H[2*N_x*(n-1) + 1 - one,  2*N_x*n     - one] += Xp
            H[2*N_x*(n-1) + 2 - one,  2*N_x*n - 1 - one] += Xn
    H += H.conj().T
    return H

def kitaev_plane_mat(N_x,N_y,mu,t_x,scp_x, pbx = 0, pby = 0, t_y=-1,scp_y=-1 ):
    # N: number of electrons
    # t: hopping constant
    # sp: superconducting pairing coefficient
    
    
    if t_y==-1:
        t_y = t_x
    if scp_y == -1:
        scp_y = scp_x

    one = 1     # will use this for changes regarding the Python indexing
    # two = 2     # count of Majorana operators (sites) SORRY IT MAKES CODE HARD TO READ, I WILL GO MAGIC
    
    L = N_x*N_y*2 # Total number of Majoranas
    M = -1j*mu/2
    Xp = 1j*(scp_x+t_x)/2
    Xn = 1j*(scp_x-t_x)/2
    Yp = 1j*(scp_y+t_y)/2
    Yn = 1j*(scp_y-t_y)/2
    
    xpdiag = M*np.ones((2*N_x-1),dtype=complex)
    xpdiag[1::2] = Xp
    xndiag = np.zeros((2*N_x-3),dtype=complex)
    xndiag[0::2] = Xn
    H_dblock = np.diag(xpdiag,-1) + np.diag(xndiag,-3)

    ypdiag = np.zeros((2*N_y*N_x-(2*N_x-1)),dtype=complex)
    ypdiag[1::2] = Yp
    yndiag = np.zeros((2*N_y*N_x-(2*N_x+1)),dtype=complex)
    yndiag[0::2] = Yn
    
    H = block_diag(*[H_dblock]*N_y)
    H += np.diag(ypdiag,-(2*N_x-1)) + np.diag(yndiag,-(2*N_x+1))

    if pbx ==1:
        pbcxpdiag = np.zeros(2*N_x*(N_y-1)+1,dtype=complex)
        pbcxpdiag[0::2*N_x] = Xp
        pbcxndiag = np.zeros(2*N_x*(N_y-1)+3,dtype=complex)
        pbcxndiag[1::2*N_x] = Xn
        H += np.diag(pbcxpdiag, 2*N_x-1) + np.diag(pbcxndiag, 2*N_x-3)
    if pby == 1:
        pbcypdiag = np.zeros(2*N_x-1,dtype=complex)
        pbcypdiag[0::2] = Yp
        pbcyndiag = np.zeros(2*N_x+1,dtype=complex)
        pbcyndiag[1::2] = Yn
        H += np.diag(pbcypdiag,2*N_x*(N_y-1)+1) + np.diag(pbcyndiag,2*N_x*(N_y-1)-1)
    H += H.conj().T
    return H

def ft_hamiltonian(H_full, cell_count_in_x, axis=0, sites=2):
    """
    Applies Fourier transform (FT) along the specified axis (0 by default) of the Hamiltonian matrix representing a 2D lattice of unit cells.

    Assumes the Hamiltonian matrix `H_full` corresponst to a 2D system with indices ordered from fastest to slowest as:
    [site index within unit cell, unit cell index in x-direction, unit cell index in y-direction].

    This function supports only 2D systems but can be extended to higher dimensions.

    Parameters
    ----------
    H_full : array_like
        The full real-space Hamiltonian matrix to be Fourier transformed.
    cell_count_in_x : integer
        Number of unit cells along the x-axis (the second fastest varying index).
    axis : int, optional
        The axis along which the FT is applied. Default is 0 (x-axis). 
        Assumes PBC holds along this chosen axis. Thus, because of translational symmetry, it is sufficient to extract on reprensentative slice along that axis and perform the Fourier transform on it.
    sites : int, optional
        Number of internal sites per unit cell. Default is 2.
    
    Returns
    -------
    H_k : ndarray
        Momentum-space Hamiltonian of shape (L_pbc, L_obc, L_obc), where:
            - L_pbc is the number of momentum points along the transformed axis,
            - L_obc is the total size of the remaining system after transformation.
    """
    N_x = cell_count_in_x
    N_y = int((H_full.shape[1])/N_x/sites)
    # print(N_y)


    H = H_full.reshape((sites, N_x, N_y, sites, N_x, N_y), order='F')
    if axis == 1:
        H = H[:,:,0,:,:,:]  # Any number should give the same result if truly periodic.
                            # PS: It would be good practice to add a periodicity check here.
        H = H.reshape((sites*N_x, sites*N_x, N_y),order = 'F')
    else:
        H = H[:,0,:,:,:,:]  # Same story as above            
        H = H.reshape((sites*N_y, sites, N_x, N_y),order = 'F')
        H = np.swapaxes(H,2,3)
        H = H.reshape((sites*N_y,sites*N_y,N_x), order = 'F')
    L_obc = H.shape[0]
    L_pbc = H.shape[-1]

    H_k = np.zeros((L_pbc, L_obc, L_obc), dtype=np.complex_)
    w_mat = np.exp(2j*np.pi*np.arange(0,1,1/L_pbc))
    w_mat = np.reshape(w_mat,(1,1,L_pbc))

    for n_pbc in range(L_pbc):
            H_k[n_pbc,:,:] = np.sum(H*np.power(w_mat,n_pbc), axis=2)
    return H_k

def ft2_hamiltonian(H_full,cell_count_in_x, sites=2):
    """
    Applies FT on both axes of the system using the Hamiltonian matrix H_full
    which represents a two-dimensional unit-cell system,
    considering that the indices of one row or column of 
    Hamiltonian are site index in unit-cell, unit-cell index in x-axis, and 
    unit-cell axis in y-axis, from fastest varying to the slowest.

    Parameters
    ----------
    H_full : array_like
        Matrix to be Fourier transformed.
    cell_count_in_x : integer
        Amount of unit cells on x-axis (second fastest varying index).
    sites : int
        Site count in unit cell
    """
    N_x = cell_count_in_x
    N_y = int((H_full.shape[1])/N_x/sites)
    H = H_full[0:sites,:]

    H = np.reshape(H,(sites,sites,N_x,N_y), order='F')

    H_k = np.zeros((N_x,N_y,sites,sites),dtype=np.complex_)
    w_x_mat = np.exp(2j*np.pi*np.arange(0,1,1/N_x))
    w_x_mat = np.reshape(w_x_mat,(1,1,N_x,1))

    w_y_mat = np.exp(2j*np.pi*np.arange(0,1,1/N_y))
    w_y_mat = np.reshape(w_y_mat,(1,1,1,N_y))

    for n_x in range(N_x):
        for n_y in range(N_y):
            H_k[n_x,n_y,:,:] = np.sum(H*np.power(w_x_mat,n_x)*np.power(w_y_mat,n_y),axis=(2,3))
    return H_k

def energy(H_full, cell_count_in_x = 0, apply_ft_on = None, sites=2):
    """
    Obtains eigenvalues of the given Hamiltonian. If 'apply_ft_on' is
    set as a valid integer, apply Fourier transform beforehand.

    Parameters
    ----------
    H_full : array_like
        Matrix to be Fourier transformed.
    cell_count_in_x : integer
        Amount of unit cells on x-axis (second fastest varying index).
    apply_ft_on : integer
        Axis on which Fourier transform will be applied. If -1, take Fourier
        transform on both axes. If None, returns the eigenvalues without 
        Fourier tranformation. 
    sites : int
        Site count in unit cell
    """
    if apply_ft_on == None:
        Hk = H_full
    else:
        N_x = cell_count_in_x
        if cell_count_in_x == 0:
            N_x = int(H_full.shape[1]/sites)
            # print('ATTENTION: cell_count_in_x is not provided, length divided by count of sites per unit cells as unit cell count.({})'.format(N_x))

        if (apply_ft_on == 0)|(apply_ft_on == 1):
            Hk = ft_hamiltonian(H_full, N_x, axis= apply_ft_on, sites=sites)
        elif (apply_ft_on == -1):
            Hk = ft2_hamiltonian(H_full, N_x, sites=sites)
        else:
            raise Exception('Invalied input for apply_ft_on')
    return np.linalg.eigvals(Hk)

def theoretical_E2(Kx,Ky,mu,t_x,scp_x, t_y=None, scp_y=None):
    """
    Returns the theoretical reciprocal space energy band of the Kitaev plane.

    Parameters
    ----------
    k_vector : array_like
        k points of energy band to be evaluated.
    mu : float
        Chemical potential of Dirac fermions.
    t : float
        Hopping constants between of n-neighbor Dirac fermions.
    scp : float
        Superconducting pairing constant of n-neighbor Dirac fermions.
    """
    if t_y==None:t_y=t_x
    if scp_y==None:scp_y=scp_x
    absE = np.sqrt( (mu/2 + t_x*np.cos(Kx) + t_y*np.cos(Ky))**2 + (scp_x*np.sin(Kx)+scp_y*np.sin(Ky))**2 )
    return  absE
# tx = 1
# m =1

# neiks = 40
# neye = 40
# cikti = abs(energy(kitaev_plane_with_for_loop(neiks,
# neye,m,tx,tx,1,1),neiks, apply_ft_on=-1,))[:,0]
# # cikti2 = np.concatenate((cikti,cikti),axis=0)
# # cikti4 = np.concatenate((cikti2,cikti2),axis=1)
# plt.plot(cikti)

# For PBC on both ends, we can plot energy bands over 4 BZs
# P = abs(energy(graphene(20,20,1,1,1,1),20,-1)[:,:,0])
# P1 = np.concatenate((P,P),axis=0)
# P2 = np.concatenate((P1,P1),axis=1)
# heatmap(P2)
# plt.show()

# %% 2D PO %%

123

# %% 2D PO mut %%

123

# %% 2D PP %%

123

# %% 2D vectors %%

def kitaev_plane(N_x,N_y,mu,t_x,scp_x, pbx = 0, pby = 0, t_y=-1,scp_y=-1 ):
    # N: number of electrons
    # t: hopping constant
    # sp: superconducting pairing coefficient
    
    
    if t_y==-1:
        t_y = t_x
    if scp_y == -1:
        scp_y = scp_x

    one = 1     # will use this for changes regarding the Python indexing
    # two = 2     # count of Majorana operators (sites) SORRY IT MAKES CODE HARD TO READ, I WILL GO MAGIC
    
    L = N_x*N_y*2 # Total number of Majoranas
    M = -1j*mu/2
    Xp = 1j*(scp_x+t_x)/2
    Xn = 1j*(scp_x-t_x)/2
    Yp = 1j*(scp_y+t_y)/2
    Yn = 1j*(scp_y-t_y)/2


    H = np.zeros((L,L),dtype=complex) # If copmlex matrix
    for m in range(one, N_x + one):
        #print(N_x)
        for n in range(one, N_y + one):
            H[2*N_x*(n-1) + 2*m     - one,  2*N_x*(n-1) + 2*m - 1 - one] += M
        for n in range(one - pby, N_y - 1 +  one):
            H[2*N_x*n     + 2*m - 1 - one,  2*N_x*(n-1) + 2*m     - one] += Yp
            H[2*N_x*n     + 2*m     - one,  2*N_x*(n-1) + 2*m - 1 - one] += Yn
    for n in range(one, N_y + one):
        for m in range(one, N_x - 1 + one):        
            H[2*N_x*(n-1) + 2*m + 1 - one,  2*N_x*(n-1) + 2*m     - one] += Xp
            H[2*N_x*(n-1) + 2*m + 2 - one,  2*N_x*(n-1) + 2*m - 1 - one] += Xn
    if pbx == 1:
        for n in range(one, N_y + one):
            H[2*N_x*(n-1) + 1 - one,  2*N_x*n     - one] += Xp
            H[2*N_x*(n-1) + 2 - one,  2*N_x*n - 1 - one] += Xn
    H += H.conj().T
    return H

def ft_hamiltonian(H_full, cell_count_in_x, axis=0, sites=2):
    """
    Applies FT on the indicated axis (0 by default) on on Hamiltonian
    matrix A which represents a two-dimensional unit-cell system,
    considering that the indices of one row or column of 
    Hamiltonian are site index in unit-cell, unit-cell index in x-axis, and 
    unit-cell axis in y-axis, from fastest varying to the slowest.

    This function only takes Hamiltonians for 2D systems but it can also be 
    expanded for more-than-two-dimensional systems.

    Parameters
    ----------
    H_full : array_like
        Matrix to be Fourier transformed.
    cell_count_in_x : integer
        Amount of unit cells on x-axis (second fastest varying index).
    axis : int
        The axis of the system on which FT will be taken. 
        PBC must be established on this axis.
    sites : int
        Site count in unit cell
    """
    N_x = cell_count_in_x
    N_y = int((H_full.shape[1])/N_x/sites)
    # print(N_y)


    H = H_full.reshape((sites, N_x, N_y, sites, N_x, N_y), order='F')
    if axis == 1:
        H = H[:,:,0,:,:,:]  # Any number should give the same result if truly periodic.
                            # PS: It would be good practice to add a periodicity check here.
        H = H.reshape((sites*N_x, sites*N_x, N_y),order = 'F')
    else:
        H = H[:,0,:,:,:,:]  # Same story as above            
        H = H.reshape((sites*N_y, sites, N_x, N_y),order = 'F')
        H = np.swapaxes(H,2,3)
        H = H.reshape((sites*N_y,sites*N_y,N_x), order = 'F')
    L_obc = H.shape[0]
    L_pbc = H.shape[-1]

    H_k = np.zeros((L_pbc, L_obc, L_obc), dtype=np.complex_)
    w_mat = np.exp(2j*np.pi*np.arange(0,1,1/L_pbc))
    w_mat = np.reshape(w_mat,(1,1,L_pbc))

    for n_pbc in range(L_pbc):
            H_k[n_pbc,:,:] = np.sum(H*np.power(w_mat,n_pbc), axis=2)
    return H_k

def ft2_hamiltonian(H_full,cell_count_in_x, sites=2):
    """
    Applies FT on both axes of the system using the Hamiltonian matrix H_full
    which represents a two-dimensional unit-cell system,
    considering that the indices of one row or column of 
    Hamiltonian are site index in unit-cell, unit-cell index in x-axis, and 
    unit-cell axis in y-axis, from fastest varying to the slowest.

    Parameters
    ----------
    H_full : array_like
        Matrix to be Fourier transformed.
    cell_count_in_x : integer
        Amount of unit cells on x-axis (second fastest varying index).
    sites : int
        Site count in unit cell
    """
    N_x = cell_count_in_x
    N_y = int((H_full.shape[1])/N_x/sites)
    H = H_full[0:sites,:]

    H = np.reshape(H,(sites,sites,N_x,N_y), order='F')

    H_k = np.zeros((N_x,N_y,sites,sites),dtype=np.complex_)
    w_x_mat = np.exp(2j*np.pi*np.arange(0,1,1/N_x))
    w_x_mat = np.reshape(w_x_mat,(1,1,N_x,1))

    w_y_mat = np.exp(2j*np.pi*np.arange(0,1,1/N_y))
    w_y_mat = np.reshape(w_y_mat,(1,1,1,N_y))

    for n_x in range(N_x):
        for n_y in range(N_y):
            H_k[n_x,n_y,:,:] = np.sum(H*np.power(w_x_mat,n_x)*np.power(w_y_mat,n_y),axis=(2,3))
    return H_k

def ground_eigensolutions(input_H, cell_count_in_x=0, sites=2):
    """
    Computes the eigenvectors corresponding to the ground state(s) of a given Hamiltonian matrix.

    This function returns all eigenvectors associated with the eigenvalue(s) that have the
    smallest absolute magnitude. These are interpreted as the ground modes, which may be 
    degenerate. The input matrix is expected to be Hermitian.

    Parameters
    ----------
    H_full : array_like
        The Hamiltonian matrix whose eigenmodes are to be computed.
        Must be square and of dimension (N, N), where N is divisible by `sites`
        (where `sites=2` by default, if not set otherwise).

    cell_count_in_x : int, optional
        Number of unit cells along the x-axis. If set to 0 (default), it is inferred that the 
        Hamiltonian represents a 1D structure, so this value is taken as the total size 
        divided by the number of sites per unit cell.

    sites : int
        Number of sites in each unit cell. Used to check consistency of matrix dimensions.

    Returns
    -------
    ground_eigvals : ndarray
        Eigenvalues (real, smallest in absolute value) associated with the ground state(s),
        sorted by absolute value.

    ground_eigvecs : ndarray
        Corresponding eigenvectors (each column is an eigenvector) of the ground state(s).

    Notes
    -----
    - Eigenvalues are sorted by their absolute value.
    - All degenerate eigenvalues within a numerical tolerance (1e-11) of the lowest absolute 
      eigenvalue are considered part of the ground eigenspace.
    - A warning is printed if any eigenvalue has a significant imaginary part.
    - The Fourier transform functionality is not yet implemented, but the structure is in place
      for future extensions to momentum-space analysis.
    """
    tol_deg = 1e-11     # Numerical tolerance for degeneracy check
    tol_imag = 1e-13    # Numberical tolerance for non-imaginary check

    if input_H.shape[1] % sites != 0:
        raise ValueError("The number of sites does not divide the matrix size evenly.")
    if cell_count_in_x == 0:
        cell_count_in_x = input_H.shape[1] // sites
    
    eigvals, eigvecs = np.linalg.eig(input_H)
    sort_indices = np.argsort(abs(eigvals), axis=-1)
    sorted_eigvals = eigvals[sort_indices]
    sorted_eigvecs = eigvecs[:, sort_indices]
    lowest_energy = abs(sorted_eigvals[0])

    # Ground states should be almost degenerate with the lowest eigenval within the numerical tolerance.
    lowmodes = np.sum( abs(sorted_eigvals) < abs(lowest_energy) + tol_deg)

    # We expect the input matrix to be Hermitian and thus eigenvalues are expected to be real!
    if sum(abs(eigvals.imag) > tol_imag):
        print("Warning: Eigenvalues have significant imaginary components.")

    return sorted_eigvals[0:lowmodes], sorted_eigvecs[:,0:lowmodes]

eigen_cache = {}
def cached_eigenstates(params):
    key = tuple(params)  # Convert list to tuple so it's hashable
    if key in eigen_cache:
        return eigen_cache[key]
    else:
        H = kitaev_plane(*params)
        gvals, gvecs = ground_eigensolutions(H, params[0])
        eigen_cache[key] = (gvals, gvecs)
        return gvals, gvecs