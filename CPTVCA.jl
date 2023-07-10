module CPTVCA

using Arpack
using ExactDiagonalization
using ExactDiagonalization: matrix
using KrylovKit
using LinearAlgebra
using QuantumLattices

export GreenFunction, InitialState, FInitialState, SInitialState, Kryvals, LehmannGreenFunction, Sysvals, VCA
export initialstate, clusterGreenFunction, differQuadraticTerms, periodization, VCAGreenFunction, singleparticlespectrum

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
function Sysvals(k::EDKind, eigensystem::Eigen, gen::OperatorGenerator, target::TargetSpace, table::Table; m::Int=200)
    gse, gs = eigensystem.values[1], eigensystem.vectors[:,1]
    setkryvals₁, setkryvals₂ = (Vector{Kryvals}(), Vector{Kryvals}())
    H₁, H₂ = matrix(expand(gen), (target[2], target[2]), table), matrix(expand(gen), (target[3], target[3]), table)
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

Calculate the cluster Green function
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
    periodization(lattice::AbstractLattice, table::Table, k::AbstractVector) -> Matrix

Carry out the periodization procedure with G-scheme
"""
function periodization(lattice::AbstractLattice, table::Table, k::AbstractVector)
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
struct VCA{L<:AbstractLattice, T<:Table, G<:OperatorGenerator, S<:Sysvals}
    lattice::L
    table::T
    origigenerator::G
    refergenerator::G
    sysvals::S
end

"""
    VCA(lattice::AbstractLattice, hilbert::Hilbert, eigensystem::Eigen, origiterms::Tuple{Vararg{Term}}, referterms::Tuple{Vararg{Term}}, target::TargetSpace, neighbors::Neighbors; m::Int=200)

Construct the Variational Cluster Approach(VCA) method for a quantum lattice system.
"""
function VCA(lattice::AbstractLattice, hilbert::Hilbert, origiterms::Tuple{Vararg{Term}}, referterms::Tuple{Vararg{Term}}, target::TargetSpace, neighbors::Neighbors; m::Int=200)
    k = EDKind(typeof(origiterms))
    table = Table(hilbert, Metric(k, hilbert))
    origibonds = bonds(lattice, neighbors)
    referbonds = filter(bond -> isintracell(bond), origibonds)
    origigenerator = OperatorGenerator(origiterms, origibonds, hilbert) 
    refergenerator = OperatorGenerator(referterms, referbonds, hilbert)
    ed = ED(lattice, hilbert, origiterms, TargetSpace(target[1])) 
    eigensystem = ExactDiagonalization.eigen(matrix(ed); nev=1)
    sysvals = Sysvals(k, eigensystem, refergenerator, target, table; m)
    return VCA{typeof(lattice), typeof(table), typeof(origigenerator), typeof(sysvals)}(lattice, table, origigenerator, refergenerator, sysvals)
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
    gf = sum((1/length(vca.lattice))*pm*gm)
    return gf
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
