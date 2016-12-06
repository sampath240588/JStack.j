                
addprocs([("iriadmin@10.63.36.22", 1), ("iriadmin@10.63.36.23", 1)])
addprocs(2)
using DataStructures, DataArrays , DataFrames, StatsFuns, GLM , Distributions, MixedModels, StatsBase, JSON, StatLib
@everywhere using DataStructures, DataArrays , DataFrames, StatsFuns, GLM , Distributions, MixedModels, StatsBase, JSON, StatLib


#function distributeDataset()
#    #jmod_fname = root*"/dfd_model.json"
#    #mod_fname
#    #ppp="/mnt/resource/analytics/models/"; run(`ls $ppp`)
#    ips = ["10.63.36.22","10.63.36.23"]
#    for ip in ips
#            cmd=`scp $root/dfd_model.* $ip:$root/`
#            println(cmd)
#            run(cmd)
#        #run(`ssh $ip mkdir $root`)
#        #run(`scp $root/dfd_model.* $ip:$root/`)
#    end
#end

root="/mnt/resource/analytics/models/Jenny-o"
root="/mnt/resource/analytics/models/NatChoice"
root="/mnt/resource/analytics/models/rev"
root="/mnt/resource/analytics/models/ALL#10"
root="/mnt/resource/analytics/models/ALL#11"
root="/mnt/resource/analytics/models/CDW#6"
root="/mnt/resource/analytics/models/HormelChili#8"
root="/mnt/resource/analytics/models/Rev#8"
                                
root="/mnt/resource/analytics/models/Jenny-o"

jmod_fname = root*"/dfd_model.json"
mod_fname = root*"/dfd_model.csv"         
modelsDict = readModels(jmod_fname) 
                


@everywhere function runGlm(mod_fname::String, jmod_fname::String, modelname::Symbol, ranef::Symbol=:empty)
    println("Loding data : ")
    dfd = readtable(mod_fname,header=true);
    modelsDict = readModels(jmod_fname)
    poolit!(dfd,modelsDict[:factors])
    m=modelsDict[modelname]
    return runGlm(dfd, m, ranef)
end

@everywhere function runGlm(dfd::DataFrame,m::Dict, ranef::Symbol=:empty)
    if ranef==:empty
        f=genF(m[:y_var],m[:finalvars])
        cols = convert(Array{Symbol},vcat(m[:finalvars], [m[:y_var], m[:logvar]]))
        println("Ok - Running ",m[:modelName]," -  Glm on proc ",myid()," with : ",f)
    else
        f=genF(m[:y_var],setdiff(vcat(m[:finalvars],ranef),[:group]))
        dfd[ranef] = relevel(dfd[ranef], "none")
        cols = convert(Array{Symbol},vcat(m[:finalvars], [m[:y_var], m[:logvar]], ranef ))
        println("Ok - Running ",m[:modelName]," - ",ranef," Glm on proc ",myid()," with : ",f)
    end
    if m[:Buyer_Pos_P1_is1]
        return glm(f, dfd[(dfd[:buyer_pos_p1].==1),cols] , m[:dist], m[:lnk])
    else
        return return glm(f, dfd[cols] , m[:dist], m[:lnk])
    end
end        
        
                

@everywhere function runGlmm(mod_fname::String, jmod_fname::String, modelname::Symbol, ranef::Symbol=:empty)
    println("Loding data : Glmm : ",modelname," ~~~ ",ranef)
    dfd = readtable(mod_fname,header=true);
    modelsDict = readModels(jmod_fname)
    poolit!(dfd,modelsDict[:factors])
    m=modelsDict[modelname]
    if ranef==:empty
        v_out=Dict()
        for r in m[:raneff]
            v_out[r] = runGlmm(dfd, m, [r])
        end
        return v_out
    else
        return runGlmm(dfd, m, [ranef])
    end
end
 

@everywhere function runGlmm(dfd::DataFrame,m::Dict, ranef::Array{Symbol}=[:empty])
    #function genFx(y::Symbol, iv::Array{Symbol},ranef::Array{Symbol})  
    #    vars=setdiff(iv,vcat([y],ranef,[:group]))
    #    eval(parse( string(y)*" ~ 1"* reduce(*, [ " + "*  string(c) for c in vars ] ) * reduce(*, [ " + "*  "(1 | "*string(c)*")" for c in ranef ] )  ) )
    #end   
    
    #function genF(y::Symbol, iv::Array{Symbol},ranef::Array{Symbol}=Symbol[])  
    #    vars=setdiff(iv,vcat([y],ranef))
    #    f = string(y)*" ~ 1"
    #    f = length(vars) > 0 ?  f * reduce(*, [ " + "*  string(c) for c in unique(vars) ] ) : f
    #    f = length(ranef) > 0 ?  f * reduce(*, [ " + "*  "(1 | "*string(c)*")" for c in unique(ranef) ] )  : f
    #    eval(parse(f))
    #end
    
    ranef = ranef==[:empty] ? m[:raneff]  : ranef
    for r in ranef
        dfd[r] = relevel(dfd[r], "none")
    end
    #setdiff(iv,vcat([y],ranef,[:group]))
    f=genF(m[:y_var],setdiff(m[:finalvars],[:group]),ranef)
    #f=genFx(m[:y_var],m[:finalvars],ranef)
    cols = convert(Array{Symbol},vcat(m[:finalvars], [m[:y_var], m[:logvar]], ranef ))
    println("Ok - Running ",m[:modelName]," - ",ranef," Glmm on proc ",myid()," with : ",f)
    if m[:Buyer_Pos_P1_is1]
        gmm1 = fit!(glmm(f, dfd[(dfd[:buyer_pos_p1].==1),cols] , m[:dist], m[:lnk]), false, 0)
        if sum(raneffect(gmm1)[:coef]) == 0 profileθ(gmm1) end
        return gmm1
    else
        gmm1 = fit!(glmm(f, dfd[cols], m[:dist], m[:lnk]), false, 0)
        if sum(raneffect(gmm1)[:coef]) == 0 profileθ(gmm1) end
        return gmm1
    end
end

  


# ******************* SEQUENTIAL *************************
function runModels(mod_fname::String, modelsDict::Dict)
    dfd = readtable(mod_fname,header=true);
    poolit!(dfd,modelsDict[:factors])
    res = OrderedDict()
    q = Symbol[]
    cnt=0
    for (k,v) in OrderedDict(m=>modelsDict[m][:raneff] for m in [:ipen,:idolocc,:iocc]) 
         for r in v push!(q,k); push!(q,r); cnt=cnt+1 end 
    end
    x=reshape(q, (2,cnt))
    ml = permutedims(x, [2, 1])
    m=:empty
    for i in 1:length(ml[:,1]) 
        if m != ml[i,:][1]
            gr=Symbol("glm_"*string(ml[i,:][1]))
            res[gr] = runGlm(dfd,modelsDict[ml[i,:][1]]) #res[gr] = runGlm(mod_fname, jmod_fname, ml[i,:][1])
            m=ml[i,:][1]
        end
        r=Symbol("glmm_"*string(ml[i,:][1])*"_"*string(ml[i,:][2]))
        res[r] = runGlmm(dfd,modelsDict[m], [ml[i,:][2]])  #res[r] = runGlmm(mod_fname, jmod_fname, ml[i,:][1], ml[i,:][2]) 
        covg=Symbol("covglm_"*string(ml[i,:][1])*"_"*string(ml[i,:][2]))
        res[covg] = runGlm(dfd,modelsDict[m], ml[i,:][2])   #res[covg] = runGlm(mod_fname, jmod_fname, ml[i,:][1], ml[i,:][2]) 
    end 
    return res                        
end        
#dx = runModels(mod_fname,modelsDict)                 
  
function consolidateResults(modelsDict::Dict,dx::OrderedDict)  
  sdf = DataFrame(parameter=String[], coef=Int64[], stderr=Int64[], zval=Int64[], pval=Int64[], model=Symbol[], ranef=Symbol[], modelType=Symbol[])
    for m in [:iocc, :idolocc, :ipen]
        g = dx[Symbol("glm_"*string(m))]
        dfx = coefDF(g,false)
        dfx[:model] = Symbol(replace(string(m),"i",""))
        dfx[:ranef] = :none
        dfx[:modelType] = :GLM
        sdf = vcat(sdf,dfx)
        grp = sdf[(sdf[:modelType].==:GLM)&(sdf[:model].==Symbol(replace(string(m),"i","")))&(sdf[:parameter].=="group"),:coef][1]
        err = sdf[(sdf[:modelType].==:GLM)&(sdf[:model].==Symbol(replace(string(m),"i","")))&(sdf[:parameter].=="group"),:stderr][1]
        for r in  modelsDict[m][:raneff]
            k="_"*string(m)*"_"*string(r)
            g = dx[Symbol("glmm"*k)]
            dfx = raneffect(g)     
            dfx[:zval]=0.0
            dfx[:pval]=0.0
            dfx[:model] = Symbol(replace(string(m),"i",""))
            dfx[:ranef]=r
            dfx[:modelType]=:GLMM
            sdf = vcat(sdf,dfx)
            g = dx[Symbol("covglm"*k)]
            dfx = coefDF(g,false)
            dfx[:model] = Symbol(replace(string(m),"i",""))
            dfx[:ranef] = r
            dfx[:modelType] = :RanCov  
            dfx = dfx[findin(dfx[:parameter],filter(x->contains(x,string(r)) ,Array(dfx[:parameter]))),:]
            #dfx[:coef]=dfx[:coef]-grp  
            #dfx[:stderr]=sqrt(dfx[:stderr].^2+err^2)
            dfx[:parameter] = map(x-> replace(x,string(r)*": ",""),dfx[:parameter])
            sdf = vcat(sdf,dfx)                    
        end
    end
    return sdf
end     
#dx = runModels(mod_fname,modelsDict) 
#dfx = consolidateResults(modelsDict, dx)      
#writetable(root*"/campaign.csv", dfx)  

 

    
    
    
    
    
               
# ******************* CONCURRENT - DEFAULT *************************                
function runModels(mod_fname::String, jmod_fname::String, modelsDict::Dict)
   q = Symbol[]
   cnt=0
   for (k,v) in OrderedDict(m=>modelsDict[m][:raneff] for m in [:ipen,:idolocc,:iocc]) 
        for r in v push!(q,k); push!(q,r); cnt=cnt+1 end 
   end
   x=reshape(q, (2,cnt))
   ml = permutedims(x, [2, 1])
   m=:empty
   for i in 1:length(ml[:,1]) 
       if m != ml[i,:][1]
           gr=Symbol("glm_"*string(ml[i,:][1]))
           @swarm gr runGlm(mod_fname, jmod_fname, ml[i,:][1])
           m=ml[i,:][1]
       end
       r=Symbol("glmm_"*string(ml[i,:][1])*"_"*string(ml[i,:][2]))
       @swarm r runGlmm(mod_fname, jmod_fname, ml[i,:][1], ml[i,:][2]) 
       covg=Symbol("covglm_"*string(ml[i,:][1])*"_"*string(ml[i,:][2])   )
       @swarm covg runGlm(mod_fname, jmod_fname, ml[i,:][1], ml[i,:][2]) 
   end 
                            
end        
runModels(mod_fname, jmod_fname,modelsDict)                   
while !iscompletew() println("Not Complete yet!"); sleep(5); end                
println("DONE!DONE!DONE!")

                    
                    
function consolidateResults(modelsDict::Dict)  
  sdf = DataFrame(parameter=String[], coef=Int64[], stderr=Int64[], zval=Int64[], pval=Int64[], model=Symbol[], ranef=Symbol[], modelType=Symbol[])
    for m in [:iocc, :idolocc, :ipen]
        g = getw(  Symbol("glm_"*string(m))    )
        #g = takew(  Symbol("glm_"*string(m))    )
        dfx = coefDF(g,false)
        dfx[:model] = Symbol(replace(string(m),"i",""))
        dfx[:ranef] = :none
        dfx[:modelType] = :GLM
        sdf = vcat(sdf,dfx)
        grp = sdf[(sdf[:modelType].==:GLM)&(sdf[:model].==Symbol(replace(string(m),"i","")))&(sdf[:parameter].=="group"),:coef][1]
        err = sdf[(sdf[:modelType].==:GLM)&(sdf[:model].==Symbol(replace(string(m),"i","")))&(sdf[:parameter].=="group"),:stderr][1]
        for r in  modelsDict[m][:raneff]
            k="_"*string(m)*"_"*string(r)
            g = getw(Symbol("glmm"*k))
            dfx = raneffect(g)     
            dfx[:zval]=0.0
            dfx[:pval]=0.0
            dfx[:model] = Symbol(replace(string(m),"i",""))
            dfx[:ranef]=r
            dfx[:modelType]=:GLMM
            sdf = vcat(sdf,dfx)
                                
            g = getw(Symbol("covglm"*k))
            dfx = coefDF(g,false)
            dfx[:model] = Symbol(replace(string(m),"i",""))
            dfx[:ranef] = r
            dfx[:modelType] = :RanCov  
            dfx = dfx[findin(dfx[:parameter],filter(x->contains(x,string(r)) ,Array(dfx[:parameter]))),:]
            #dfx[:coef]=dfx[:coef]-grp  
            #dfx[:stderr]=sqrt(dfx[:stderr].^2+err^2)
            dfx[:parameter] = map(x-> replace(x,string(r)*": ",""),dfx[:parameter])
            sdf = vcat(sdf,dfx)                    
        end
    end
    return sdf
end     
dfx = consolidateResults(modelsDict)  
writetable(root*"/campaign.csv", dfx)                      
        
        
                
  
                      