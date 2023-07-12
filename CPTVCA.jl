module CPTVCA

using Arpack
using ExactDiagonalization
using ExactDiagonalization: matrix
using KrylovKit
using LinearAlgebra
using QuantumLattices

export GreenFunction, InitialState, FInitialState, SInitialState, Kryvals, LehmannGreenFunction, Sysvals, Perioder, VCA
export initialstate, clusterGreenFunction, differQuadraticTerms, periodmatrix, VCAGreenFunction, singleparticlespectrum

"""
    abstract type GreenFunction
"""
abstract type GreenFunction end

"""
    abstract type InitialState

The Operators needed in calculating the initial vector in Lanczos iteration method of krylov subspace
"""
abstract type InitialState end

"""
    FInitialState{O<:Operator} <: InitialState
    SInitialState{O<:Operator} <: InitialState

The Operators in a fermion/spin system needed in calculating the initial vector in Lanczos iteration method of krylov subspace 
"""
struct FInitialState{O<:Operator} <: InitialState
    sign::Int
    ops::OperatorSum{O}
    function FInitialState(sign::Int, key::Tuple)
        ops = Operators(1*CompositeIndex(Index(key[2], FID{:f}(key[3], key[1], sign)), [0.0, 0.0], [0.0, 0.0]))
        new{eltype(ops)}(sign, ops)
    end
end
struct SInitialState <: InitialState
    sign::Int
    key::Tuple
    function SInitialState(sign::Int, key::Tuple)
        new(sign,key)
        #to be update
    end
end

"""
    initialstate(::EDKind{:FED}, sign::Int, key::Tuple) -> FInitialState
    initialstate(::EDKind{:SED}, sign::Int, key::Tuple) -> SInitialState

Get the Operators needed in calculating the initial vector
"""
function initialstate(::EDKind{:FED}, sign::Int, key::Tuple) FInitialState(sign, key) end
function initialstate(::EDKind{:SED}, sign::Int, key::Tuple) SInitialState(sign, key) end

"""
    Kryvals{M<:AbstractMatrix, R<:Real, V<:AbstractVector}

The information obtained with the krylov subspace method that needed in calculating the cluster Green function
"""
struct Kryvals{M<:AbstractMatrix, R<:Real, V<:AbstractVector}
    tridimatrix::M
    norm::R
    projectvector::V
    function Kryvals(matrix::AbstractMatrix, sstate::AbstractVector, m::Int=200)
        krybasis, tridimatrix = genkrylov(matrix, sstate, m)
        projectvector = KrylovKit.project!(zeros(Complex, m), krybasis, sstate)
        norm = √(sstate'*sstate)
        new{typeof(tridimatrix), typeof(norm), typeof(projectvector)}(tridimatrix, norm, projectvector)
    end
end

"""
    genkrylov(matrix::AbstractMatrix, sstate::AbstractVector, m::Int)

Generate a krylov subspace with Lanczos iteration method
"""
function genkrylov(matrix::AbstractMatrix, sstate::AbstractVector, m::Int)
    matrix = Hermitian(matrix)
    orth = KrylovKit.ModifiedGramSchmidt()
    iterator = LanczosIterator(matrix, sstate, orth)
    factorization = KrylovKit.initialize(iterator)
    for _ in 1:m-1
        KrylovKit.expand!(iterator, factorization)
    end
    basis_vectors = basis(factorization)
    T = rayleighquotient(factorization)
    return basis_vectors, T
end

"""
    LehmannGreenFunction{R<:Real, I<:Int, S<:Kryvals} <: GreenFunction

The minimum element of a Green function in Lehmann representation, e.g. <<c_{im↑}|c†_{jn↓}>>. the field sign is used to indicate 
whether it is the retarded part(sign is 2) or the advanced part(sign is 1) of a Causal Green function.
"""
struct LehmannGreenFunction{R<:Real, I<:Int, S<:Kryvals} <: GreenFunction
    GSEnergy::R
    sign::I
    kryvals_l::S
    kryvals_r::S
    function LehmannGreenFunction(gse::Real, sign::Int, kryvals_l::Kryvals, kryvals_r::Kryvals)
        new{typeof(gse), typeof(sign), typeof(kryvals_l)}(gse, sign, kryvals_l, kryvals_r)
    end
end
function (gf::LehmannGreenFunction)(ω::Real, μ::Real; η::Real=0.05)
    Im = Matrix{Complex}(I, size(gf.kryvals_l.tridimatrix,2), size(gf.kryvals_l.tridimatrix,2))
    if gf.sign == 1
        lgf = dot(gf.kryvals_l.projectvector, inv((ω + η*im + μ - gf.GSEnergy)*Im + Matrix(gf.kryvals_r.tridimatrix))[:, 1])*gf.kryvals_r.norm
    elseif gf.sign == 2
        lgf = dot(gf.kryvals_l.projectvector, inv((ω + η*im + μ + gf.GSEnergy)*Im - Matrix(gf.kryvals_r.tridimatrix))[:, 1])*gf.kryvals_r.norm
    end
    return lgf
end

"""
    Sysvals{K<:EDKind, R<:Real, S<:Kryvals}

The all information needed to calculate the Green Function of a finite size system 
"""
struct Sysvals{K<:EDKind, R<:Real, S<:Kryvals}
    GSEnergy::R
    setkryvals₁::Vector{S}
    setkryvals₂::Vector{S}
end
function Sysvals(k::EDKind, eigensystem::Eigen, ops::OperatorSum, target::TargetSpace, table::Table; m::Int=200)
    gse, gs = eigensystem.values[1], eigensystem.vectors[:,1]
    setkryvals₁, setkryvals₂ = (Vector{Kryvals}(), Vector{Kryvals}())
    H₁, H₂ = matrix(ops, (target[2], target[2]), table), matrix(ops, (target[3], target[3]), table)
    orderkeys = sort(collect(keys(table)), by = x -> table[x])
    for key in orderkeys
        ops₁, ops₂ = initialstate(k, 1, key).ops, initialstate(k, 2, key).ops
        sstate₁, sstate₂ = (matrix(ops₁, (target[2], target[1]), table)*gs)[:,1], (matrix(ops₂, (target[3], target[1]), table)*gs)[:,1]
        push!(setkryvals₁, Kryvals(H₁, sstate₁, m))
        push!(setkryvals₂, Kryvals(H₂, sstate₂, m))
    end
    return Sysvals{typeof(k), typeof(gse), eltype(setkryvals₁)}(gse, setkryvals₁, setkryvals₂)
end

"""
    clusterGreenFunction(sys::Sysvals, ω::Real, μ::Real) -> Matrix

Calculate the cluster Green function with ED solver
"""
function clusterGreenFunction(sys::Sysvals, ω::Real, μ::Real)
    cgfm = zeros(Complex, length(sys.setkryvals₁), length(sys.setkryvals₁))
    for i in 1:length(sys.setkryvals₁), j in 1:length(sys.setkryvals₁)
        gf₂ = LehmannGreenFunction(sys.GSEnergy, 2, sys.setkryvals₂[i], sys.setkryvals₂[j])
        gf₁ = LehmannGreenFunction(sys.GSEnergy, 1, sys.setkryvals₁[j], sys.setkryvals₁[i])
        cgfm[i, j] += gf₂(ω, μ) + gf₁(ω, μ)
    end
    return cgfm
end

"""
    differQuadraticTerms(oops::OperatorSum, rops::OperatorSum, table::Table, k::AbstractVector) -> Matrix

Calculate the difference between the Hamiltonian's quadratic terms of the original system and a reference system
"""
function differQuadraticTerms(ogen::OperatorGenerator, rgen::OperatorGenerator, table::Table, k::AbstractVector)
    om, rm = (zeros(Complex, length(table), length(table)), zeros(Complex, length(table), length(table)))
    oops, rops = filter(op -> length(op) == 2, collect(expand(ogen))), filter(op -> length(op) == 2, collect(expand(rgen)))
    for oop in oops
        seq₁, seq₂ = table[oop[1].index'], table[oop[2].index]
        phase = isapprox(norm(icoordinate(oop)), 0.0) ? one(eltype(om)) : convert(eltype(om), 2*exp(im*dot(k, icoordinate(oop))))
        om[seq₁, seq₂] += oop.value*phase
    end
    for rop in rops 
        seq₁, seq₂ = table[rop[1].index'], table[rop[2].index]
        rm[seq₁, seq₂] += rop.value
    end
    return om - rm
end


"""
    The solver to calculate the cluster Green function
"""
abstract type CGFsolver end
"""
    The ED solver to calculate the cluster Green function
"""
struct EDsolver{S<:Sysvals} <: CGFsolver 
    sysvals::S
    function EDsolver(hilbert::Hilbert, referterms::Tuple{Vararg{Term}}, target::TargetSpace;m::Int=200)
        k = EDKind(typeof(referterms))
        table = Table(hilbert, Metric(k, hilbert))
        rops = expand(OperatorGenerator(referterms, referbonds, hilbert; table))
        Hₘ = matrix(rops, (target[1], target[1]), table)
        eigens = eigs(Hₘ; nev=1, which=:SR, tol=0.0,maxiter=300,  sigma=nothing, ritzvec=true, v0=[])
        eigensystem = Eigen(eigens[1], eigens[2])
        sysvals = Sysvals(k, eigensystem, rops, target, table; m)
        new{typeof(sysvals)}(sysvals)
    end
end

"""
    Perioder{I<:AbstractVector}
    Perioder(::EDKind{:FED}, cluster::AbstractLattice, unitcell::AbstractLattice)

User should ensure that the cluster you choosed is compatible with the lattice generated by the unitcell you input and the unitcell you input should be enclosed in the cluster you choosed sharing an original point with the cluster.

"""
struct Perioder{I<:AbstractVector}
    per::Vector{I}
end

function Perioder(::EDKind{:FED}, cluster::AbstractLattice, unitcell::AbstractLattice)
    @assert !isempty(unitcell.vectors) "the vectors in unitcell cannot be empty !"
    hilbert₁, hilbert₂ = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster)), Hilbert(site=>Fock{:f}(1, 2) for site=1:length(unitcell))
    table₁, table₂ = Table(hilbert₁, Metric(EDKind{:FED}(), hilbert₁)), Table(hilbert₂, Metric(EDKind{:FED}(), hilbert₂))
    seq₁, seq₂ =  sort(collect(keys(table₁)), by = x -> table₁[x]), sort(collect(keys(table₂)), by = x -> table₂[x])
    per = [Vector{Int}() for _ in 1:length(seq₂)] 
    for i in 1:length(seq₂)
        for j in 1:length(seq₁)
            if seq₁[j][1]==seq₂[i][1] && seq₁[j][3]==seq₂[i][3]
                if issubordinate(cluster.coordinates[:,seq₁[j][2]]-unitcell.coordinates[:,seq₂[i][2]], unitcell.vectors) 
                    push!(per[i], j)
                end
            end
        end
    end
    return Perioder{eltype(per)}(per)
end


"""
    periodmatrix(lattice::AbstractLattice, table::Table, k::AbstractVector) -> Matrix

Carry out the periodmatrix procedure with G-scheme
"""
function periodmatrix(lattice::AbstractLattice, table::Table, k::AbstractVector)
    orderkeys = sort(collect(keys(table)), by = x -> table[x])
    coordinates = Vector{Vector}()
    pm = zeros(Complex, length(table), length(table))
    for key in orderkeys
        push!(coordinates, lattice.coordinates[:, key[2]])
    end
    for i in eachindex(coordinates), j in eachindex(coordinates)
        pm[i, j] = exp(-im*dot(k, (coordinates[i] - coordinates[j])))
    end
    return pm
end



"""
    VCA{K<:EDKind, L<:AbstractLattice, T<:Table, G<:OperatorGeneratore, S<:Sysvals}

Variational Cluster Approach(VCA) method for a quantum lattice system.
"""
struct VCA{L<:AbstractLattice, G<:OperatorGenerator, S<:Solver, P<:Perioder}
    cluster::L
    unitcell::L
    origigenerator::G
    refergenerator::G
    solver::S
    per::P
end


"""
    VCA(lattice::AbstractLattice, hilbert::Hilbert, eigensystem::Eigen, origiterms::Tuple{Vararg{Term}}, referterms::Tuple{Vararg{Term}}, target::TargetSpace, neighbors::Neighbors; m::Int=200)

Construct the Variational Cluster Approach(VCA) method for a quantum lattice system.
"""
function VCA(solver::Solver, cluster::AbstractLattice, unitcell::AbstractLattice, hilbert::Hilbert, origiterms::Tuple{Vararg{Term}}, referterms::Tuple{Vararg{Term}}, target::TargetSpace, neighbors::Neighbors; m::Int=200)
    k = EDKind(typeof(origiterms))
    table = Table(hilbert, Metric(k, hilbert))
    origibonds = bonds(cluster, neighbors)
    referbonds = filter(bond -> isintracell(bond), origibonds)
    origigenerator, refergenerator = OperatorGenerator(origiterms, origibonds, hilbert; table), OperatorGenerator(referterms, referbonds, hilbert; table) 
    perioder = Perioder(k, cluster, unitcell)
    return VCA{typeof(cluster), typeof(origigenerator), typeof(solver), typeof(perioder)}(cluster, unitcell, origigenerator, refergenerator, solver, perioder)
end

"""
Calculate the Causal Green Function with VCA method in k-ω space
"""
function VCAGreenFunction(vca::VCA, k::AbstractVector, ω::Real, μ::Real) <: GreenFunction
    vm = differQuadraticTerms(vca.origigenerator, vca.refergenerator, vca.table, k)
    Im = Matrix{Complex}(I, size(vm, 1), size(vm, 2))
    cm = clusterGreenFunction(vca.sysvals, ω, μ)
    gm = cm*inv(Im - vm*cm)
    pm = periodization(vca.lattice, vca.table, k)
    gfm = pm*gm
    pgfm = zeros(Complex, length(vca.perioder),length(vca.perioder))
    for i in 1:length(vca.perioder.per), j in 1:length(vca.perioder.per)
        pgfm[i, j] = (1/length(vca.perioder.per[i]))*sum(gfm[vca.perioder.per[i], vca.perioder.per[j]])
    end
    return gfm
end 

"""
Construct the k-ω matrix to store the data of single particle spectrum
"""
function singleparticlespectrum(vca, path, range, μ)
    A = zeros(Float64, length(range), length(path[1]))
    function calculate_element(m, i)
        k = path[1][i]
        ω = range[m]
        return (-1 / π) * imag(VCAGreenFunction(vca, k, ω, μ))
    end
    for i in eachindex(path[1])
        for m in eachindex(range)
            A[m, i] = calculate_element(m, i)
        end
    end
    return A
end










end # module CPTVCA
