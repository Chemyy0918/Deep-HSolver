# File: HSolver.jl                              
# Author: Yanmin Qian                        
# Date: 2025-11-13                               
# Description: 
# 求解DeepH输出的哈密顿矩阵
# 修改自DeepH开发组的sparse_calc.jl
#导库区
using Distributed #分布式并行库
@everywhere using LinearAlgebra #线性代数库
@everywhere using SparseArrays #稀疏矩阵库
using HDF5  
using JSON
using ArgParse #命令行配置库
using DelimitedFiles #分隔符文本库
start_time=time()
#常量区
@everywhere default_dtype = Complex{Float64}
#函数区
function parse_commandline() #命令行配置函数
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--input_dir", "-i"
            help = "path of rlat.dat, orbital_types.dat, site_positions.dat, hamiltonians_dft.h5, and overlaps.h5"
            arg_type = String
            default = "./"
        "--config"
            help = "config file in the format of JSON"
            arg_type = String
    end
    return parse_args(s)
end

function _create_dict_h5(filename::String) #h5转字典
    fid = h5open(filename, "r")
    T = eltype(fid[keys(fid)[1]])
    d_out = Dict{Array{Int64,1}, Array{T, 2}}()
    for key in keys(fid)
        data = read(fid[key])
        nk = map(x -> parse(Int64, convert(String, x)), split(key[2 : length(key) - 1], ','))
        d_out[nk] = permutedims(data)
    end
    close(fid)
    return d_out
end

@everywhere function HSolver(kdata) #kdata格式:[kx,ky,kz]
    kx = kdata[1]
    ky = kdata[2]
    kz = kdata[3]
    H_k = spzeros(default_dtype, norbits, norbits)
    S_k = spzeros(default_dtype, norbits, norbits)
    for R in keys(H_R)
        H_k += H_R[R] * exp(im*2π*([kx, ky, kz]⋅R))
        S_k += S_R[R] * exp(im*2π*([kx, ky, kz]⋅R))
    end
    H_dense = Matrix(H_k)
    S_dense = Matrix(S_k) #CSC稀疏矩阵稠密化
    H_dense=(H_dense+H_dense')/2
    S_dense=(S_dense+S_dense')/2 #严格厄米化
    if spinful==true
        N = size(S_dense, 1) ÷ 2
        Suu = S_dense[1:N, 1:N]
        Sud = S_dense[1:N, N+1:2N]
        Sdu = S_dense[N+1:2N, 1:N]       
        Sdd = S_dense[N+1:2N, N+1:2N]
        σ_x = zeros(ComplexF64, norbits)
        σ_y = zeros(ComplexF64, norbits)
        σ_z = zeros(ComplexF64, norbits)
    end
    ε,Φ = eigen(H_dense, S_dense) #核心求解步骤HΦ=εSΦ
    for i in 1:norbits
        φ = Φ[:, i]                   
        φ = φ / norm(φ)
        if spinful==true #处理自旋磁矩期望         
            φ_u = φ[1:size(Φ, 1) ÷ 2] 
            φ_d = φ[size(Φ, 1) ÷ 2 + 1:end]
            σ_x[i] = dot(φ_u,Sud*φ_d)+dot(φ_d,Sdu*φ_u)
            σ_y[i] = -im*dot(φ_u,Sud*φ_d)+im*dot(φ_d,Sdu*φ_u)
            σ_z[i] = dot(φ_u,Suu*φ_u)-dot(φ_d,Sdd*φ_d)
        end
    end
    if spinful==true
        return [real(ε),real(σ_x),real(σ_y),real(σ_z)]
    else
        return [real(ε)]
    end
end

function FDD(E,Ef,T) #费米-狄拉克分布
    kB = 8.617333*10^(-5)
    if spinful==true
        return 1/(exp((E-Ef)/(kB*T))+1)
    else
        return 2/(exp((E-Ef)/(kB*T))+1) #一个能级放两个电子故乘2
    end
end
#主函数
##读取文件
###从命令行读取config.json
parsed_args = parse_commandline()
config = JSON.parsefile(parsed_args["config"])
###从当前文件夹读取info.json,即读取spinful
if isfile(joinpath(parsed_args["input_dir"],"info.json"))
    spinful = JSON.parsefile(joinpath(parsed_args["input_dir"],"info.json"))["isspinful"]
else
    spinful = false
end
@everywhere spinful=$spinful
###从当前文件夹读取原子坐标并处理
site_positions = readdlm(joinpath(parsed_args["input_dir"], "site_positions.dat"))
nsites = size(site_positions, 2)
###从当前文件夹读取轨道信息
orbital_types_f = open(joinpath(parsed_args["input_dir"], "orbital_types.dat"), "r")
site_norbits = zeros(nsites)
orbital_types = Vector{Vector{Int64}}()
for index_site = 1:nsites
    orbital_type = parse.(Int64, split(readline(orbital_types_f)))
    push!(orbital_types, orbital_type)
end
site_norbits = (x->sum(x .* 2 .+ 1)).(orbital_types) * (1 + spinful)
norbits = sum(site_norbits)
@everywhere norbits = $norbits
site_norbits_cumsum = cumsum(site_norbits)
###从当前文件夹读取倒格子信息
rlat = readdlm(joinpath(parsed_args["input_dir"], "rlat.dat"))
###从当前文件夹读取价电子信息
element = Int.(readdlm(joinpath(parsed_args["input_dir"], "element.dat")))
val_element=config["val_atom"]
global val=0
for i in element
    for j in val_element
        if i==j[1]
            global val+=j[2]
        end
    end
end
println("总价电子数为$val")
###读取两个h5文件
hamiltonians = _create_dict_h5(joinpath(parsed_args["input_dir"], "hamiltonians_dft.h5"))
overlaps = _create_dict_h5(joinpath(parsed_args["input_dir"], "overlaps.h5"))
println("读取配置完成")
##矩阵构造
###声明数组
I_R = Dict{Vector{Int64}, Vector{Int64}}()
J_R = Dict{Vector{Int64}, Vector{Int64}}()
H_V_R = Dict{Vector{Int64}, Vector{default_dtype}}()
S_V_R = Dict{Vector{Int64}, Vector{default_dtype}}()
H_R = Dict{Vector{Int64}, SparseMatrixCSC{default_dtype, Int64}}()
S_R = Dict{Vector{Int64}, SparseMatrixCSC{default_dtype, Int64}}()
###构造COO稀疏矩阵
for key in collect(keys(hamiltonians))
    hamiltonian = hamiltonians[key]
    if (key ∈ keys(overlaps))
        overlap = overlaps[key]
    else
        overlap = zero(hamiltonian)
    end
    if spinful==true
        overlap = vcat(hcat(overlap,zeros(size(overlap))),hcat(zeros(size(overlap)),overlap))
    end
    R = key[1:3]
    atom_i=key[4]
    atom_j=key[5]
    if !(R ∈ keys(I_R))
        I_R[R] = Vector{Int64}()
        J_R[R] = Vector{Int64}()
        H_V_R[R] = Vector{default_dtype}()
        S_V_R[R] = Vector{default_dtype}()
    end
    for block_matrix_i in 1:site_norbits[atom_i] 
        for block_matrix_j in 1:site_norbits[atom_j] 
            coo_i = site_norbits_cumsum[atom_i] - site_norbits[atom_i] + block_matrix_i
            coo_j = site_norbits_cumsum[atom_j] - site_norbits[atom_j] + block_matrix_j
            push!(I_R[R], coo_i)
            push!(J_R[R], coo_j)
            push!(H_V_R[R], hamiltonian[block_matrix_i, block_matrix_j])
            push!(S_V_R[R], overlap[block_matrix_i, block_matrix_j])
        end
    end
end
###构造CSC矩阵
for R in keys(I_R)
    H_R[R] = sparse(I_R[R], J_R[R], H_V_R[R], norbits, norbits)
    S_R[R] = sparse(I_R[R], J_R[R], S_V_R[R], norbits, norbits)
end
@everywhere H_R = $H_R
@everywhere S_R = $S_R #把构造完成的字典广播到所有进程
println("矩阵构造完成")
##k网格上的矩阵求解
###求解参数读取
###声明最终容器
T=config["Temperature"]
MP=config["kpoints"]
k_data=[]
for a in 1:MP[1]
    for b in 1:MP[2]
        for c in 1:MP[3]
            push!(k_data,[(2*a-MP[1]-1)/(2*MP[1]),(2*b-MP[2]-1)/(2*MP[2]),(2*c-MP[3]-1)/(2*MP[3])])
        end
    end
end
ε_band = zeros(Float64, norbits, length(k_data))
if spinful==true
    σ_x_band = zeros(Float64, norbits, length(k_data))
    σ_y_band = zeros(Float64, norbits, length(k_data))
    σ_z_band = zeros(Float64, norbits, length(k_data))
end
###求解主派发
println("正在计算k网格...")
results=pmap(HSolver,k_data)
println("k网格上所有k点求解完成")
##读取返回并保存
for i in 1:length(k_data)
    ε_band[:,i]=results[i][1]
    if spinful==true
        σ_x_band[:,i]=results[i][2]
        σ_y_band[:,i]=results[i][3]
        σ_z_band[:,i]=results[i][4]
    end
end
#writedlm("Energy.txt",ε_band)
#writedlm("K-points.txt",k_data)
#if spinful==true
#    writedlm("sigma_x.txt",σ_x_band)
#    writedlm("sigma_y.txt",σ_y_band)
#    writedlm("sigma_z.txt",σ_z_band)
#end
println("k网格计算结果写入完成")
##二分法迭代求解费米能级
global E1=minimum(ε_band)
global E2=maximum(ε_band)
global Ef=0
while(abs(E1-E2)>10^(-3))
    global Ef=(E1+E2)/2
    cnt=0
    for n in 1:norbits
        for k in 1:length(k_data)
            cnt+=1/length(k_data)*FDD(ε_band[n,k],Ef,T)
        end
    end
    if cnt < val
        global E1=Ef
    else
        global E2=Ef
    end
end
###求总能量
global total_energy=0
for n in 1:norbits
    for k in 1:length(k_data)
        global total_energy+=1/length(k_data)*FDD(ε_band[n,k],Ef,T)*ε_band[n,k]
    end
end
kpath=config["kpath"]
k_path_site = []
###设计k路径
for l in 1:length(kpath)
    n_kpath = kpath[l][1]
    kxi, kxf = kpath[l][2], kpath[l][5]
    kyi, kyf = kpath[l][3], kpath[l][6]
    kzi, kzf = kpath[l][4], kpath[l][7]
    kxpath = LinRange(kxi, kxf, n_kpath)
    kypath = LinRange(kyi, kyf, n_kpath)
    kzpath = LinRange(kzi, kzf, n_kpath)
    kmat = hcat(kxpath, kypath, kzpath)
    for i in 1:n_kpath
        push!(k_path_site,kmat[i,:])
    end
end
ε_band_path = zeros(Float64,norbits, length(k_path_site))
if spinful==true
    σ_x_band_path = zeros(Float64, norbits, length(k_path_site))
    σ_y_band_path = zeros(Float64, norbits, length(k_path_site))
    σ_z_band_path = zeros(Float64, norbits, length(k_path_site))
end
###能带主派发
println("正在计算能带...")
results_path=pmap(HSolver,k_path_site)
println("能带计算完成")
for i in 1:length(k_path_site)
    ε_band_path[:,i]=results_path[i][1]
    if spinful==true
        σ_x_band_path[:,i]=results_path[i][2]
        σ_y_band_path[:,i]=results_path[i][3]
        σ_z_band_path[:,i]=results_path[i][4]
    end
end
writedlm("Energy.txt",ε_band_path)
#writedlm("K-PATH.txt",k_path_site)
kp_normed=[0.0]
for j in 2:length(k_path_site)
    push!(kp_normed,norm(k_path_site[j][1]*rlat[:,1]+k_path_site[j][2]*rlat[:,2]+k_path_site[j][3]*rlat[:,3]-k_path_site[j-1][1]*rlat[:,1]-k_path_site[j-1][2]*rlat[:,2]-k_path_site[j-1][3]*rlat[:,3]))
end
kp_normed=cumsum(kp_normed)
kp_normed=kp_normed/maximum(kp_normed)
writedlm("K-PATH.txt",kp_normed)
if spinful==true
    writedlm("sigma_x.txt",σ_x_band_path)
    writedlm("sigma_y.txt",σ_y_band_path)
    writedlm("sigma_z.txt",σ_z_band_path)
end
println("能带计算结果写入结束")
println("任务结束")
end_time=time()
elapsed=end_time-start_time
println("总耗时:$elapsed s")
println("费米能级:$Ef eV")
println("总能量:$total_energy eV")
