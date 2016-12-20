
using DataStructures, DataArrays , DataFrames, StatsFuns, GLM , Distributions, MixedModels, StatsBase, JSON, StatLib, JuMP, NLopt, NLsolve # JStack,
root="/mnt/resource/analytics/models/rev"
root="/mnt/resource/analytics/models/ALL#10"
jmod_fname = root*"/dfd_model.json"
mod_fname = root*"/dfd_model.csv"
modelsDict = readModels(jmod_fname) 
modelsDict[:occ]=modelsDict[:iocc]; delete!(modelsDict,:iocc)
modelsDict[:dolocc]=modelsDict[:idolocc]; delete!(modelsDict,:idolocc)
modelsDict[:pen]=modelsDict[:ipen]; delete!(modelsDict,:ipen)
campaign_fname = root*"/campaign.csv" 
dfx = readtable(campaign_fname,header=true);
df_data = readtable(mod_fname,header=true);  

dfx[:unadj_mean_score0] = 0.0
dfx[:unadj_mean_score1] = 0.0
dfx[:adj_mean_score0 ] = 0.0
dfx[:adj_mean_score1 ] = 0.0
dfx[:unadj_avg_expsd_hh_pre]=0.0
dfx[:unadj_avg_expsd_hh_pst]=0.0
dfx[:unadj_avg_cntrl_hh_pre]=0.0
dfx[:unadj_avg_cntrl_hh_pst]=0.0
dfx[:unadj_avg_cntrl_hh_pre]=0.0
dfx[:unadj_avg_cntrl_hh_pst]=0.0
dfx[:onetail_80_pct_intrvl_lb]=0.0
dfx[:onetail_80_pct_intrvl_ub]=0.0
dfx[:onetail_90_pct_intrvl_lb]=0.0
dfx[:onetail_90_pct_intrvl_ub]=0.0
dfx[:twotail_80_pct_intrvl_lb]=0.0
dfx[:twotail_80_pct_intrvl_ub]=0.0
dfx[:twotail_90_pct_intrvl_lb]=0.0
dfx[:twotail_90_pct_intrvl_ub]=0.0
dfx[:onetail_pval] = 0.0
dfx[:twotail_pval] = 0.0
dfx[:M]=0
dfx[:Mt]=0
dfx[:Mc]=0
dfx[:N]=0
dfx[:Nt]=0
dfx[:Nc]=0
# ************ EQUATIONS COLS *****************
#dfx[:unadj_mean_score0_eq] = ""
#dfx[:unadj_mean_score1_eq] = ""
dfx[:adj_mean_score0_eq ] = ""
dfx[:adj_mean_score1_eq ] = ""
#dfx[:unadj_avg_expsd_hh_pre_eq]=""
#dfx[:unadj_avg_expsd_hh_pst_eq]=""
#dfx[:unadj_avg_cntrl_hh_pre_eq]=""
#dfx[:unadj_avg_cntrl_hh_pst_eq]=""
#dfx[:unadj_avg_cntrl_hh_pre_eq]=""
#dfx[:unadj_avg_cntrl_hh_pst_eq]=""


# Anonymous function
inDFX(c::Symbol) = c in names(dfx)
mlist = [modelsDict[:occ],modelsDict[:dolocc],modelsDict[:pen]]  # otherwise we'd include modelsDict[:factors]...etc

function adjustDFX(dfx::DataFrame)
    dfx[:adj_coef] = 0.0
    dfx[:adj_stderr]=  0.0
    noneB(model::String,ranef::String) =  dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==model)&(dfx[:ranef].==ranef),:coef][1]
    noneStdErr(model::String,ranef::String) =  dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==model)&(dfx[:ranef].==ranef),:stderr][1]
    grp(model::String)= dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==model)&(dfx[:parameter].=="group"),:coef][1]
    grp_stderr(model::String)= dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==model)&(dfx[:parameter].=="group"),:stderr][1]
    for m in mlist 
        mname=string(m[:modelName])
        g=grp(mname)
        se=grp_stderr(mname)
        for r in m[:raneff]
            rs=string(r)        
            dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==mname)&(dfx[:ranef].==rs)&(dfx[:parameter].!="none"),:adj_coef]=dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==mname)&(dfx[:ranef].==rs)&(dfx[:parameter].!="none"),:coef]-noneB(mname,rs)
            m = mean(dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==mname)&(dfx[:ranef].==rs)&(dfx[:parameter].!="none"),:adj_coef])
            f=g/m
            dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==mname)&(dfx[:ranef].==rs)&(dfx[:parameter].!="none"),:adj_coef]=dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==mname)&(dfx[:ranef].==rs)&(dfx[:parameter].!="none"),:adj_coef]*f
            
            dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==mname)&(dfx[:ranef].==rs)&(dfx[:parameter].!="none"),:adj_stderr]= sqrt.(dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==mname)&(dfx[:ranef].==rs)&(dfx[:parameter].!="none"),:stderr].^2 + grp_stderr(mname)^2 )   #+ noneStdErr(mname,rs)^2    
        end
    end
    dfx[(dfx[:modelType].=="GLMM"),:zval] = dfx[(dfx[:modelType].=="GLMM"),:adj_coef] ./ dfx[(dfx[:modelType].=="GLMM"),:adj_stderr] 
    dfx[(dfx[:modelType].=="GLMM"),:pval] = 2.0 .* ccdf(Normal(), abs(  dfx[(dfx[:modelType].=="GLMM"),:zval]  ))  
    return dfx
end
dfx = adjustDFX(dfx)


function genRandCols(dfx::DataFrame, dfd::DataFrame)   #Creates cols with levels replaced with coef :ranef_occ :ranef_dolocc :ranef_pen 
    for m in mlist
        model=string(m[:modelName])
        modelType="GLMM"
        for r in m[:raneff]
            sr=string(r)
            rs=Symbol(sr*"_"*model)
            println(rs)  
            dfd[rs]=deepcopy(dfd[r])
            for row in eachrow(dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model)&(dfx[:ranef].==sr) ,:])
                dfd[(dfd[rs].==row[:parameter]),rs] = string(row[:coef])
                println(rs," ~~ ",row[:parameter])
            end
            if m[:Buyer_Pos_P1_is1] dfd[dfd[:buyer_pos_p1].==0,rs] = "0.0" end
            dfd[rs] = map(x->parse(Float64,x),dfd[rs])
            dfd[rs] = convert(Array{Float64},dfd[rs]) 
        end
    end
end
genRandCols(dfx, df_data)

dfx = dfx[(dfx[:parameter].*dfx[:modelType].!="noneGLMM"),:]  # Remove NONE - here cause need none when gen Rand Cols - convert Float!!!!

function genFixedEQUASION(dfx::DataFrame, m::Dict, con::String) # conditional equations 
    model=string(m[:modelName])
    modelType="GLM"
    dfname="df_data"
    adj0 = replace(replace(con,"XYZ","0"),"dfname",dfname)
    adj1 = replace(replace(con,"XYZ","1"),"dfname",dfname)
    println(dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model) ,:])
    intercept=dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model)&(dfx[:parameter].=="(Intercept)") ,[:coef]][1][1]
    grp=dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model)&(dfx[:parameter].=="group") ,[:coef]][1][1]
    vout0=string(intercept)*""
    vout1=string(intercept)*""
    results=dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model) ,[:parameter,:coef]]
    for row in eachrow( results[findin(results[:parameter],map(x->string(x),    setdiff(m[:finalvars],[:group])  )) ,[:parameter,:coef]] )
        vout0=vout0*"+("*dfname*"[$adj0:"*string(row[:parameter])*"].*"*string(row[:coef])*")"
        vout1=vout1*"+("*dfname*"[$adj1:"*string(row[:parameter])*"].*"*string(row[:coef])*")"
    end
    vout1=vout1*"+"*string(grp)
    return vout0, vout1
end


function genFixedCols(dfx::DataFrame,df_data::DataFrame,modelsDict::Dict)   # Creates :occ0 :occ1 :dolocc0 :dolocc1 :pen0 :pen1
    for m in mlist
        r0,r1=genFixedEQUASION(dfx,m,"")  #no conditions = populate whole dataset col - even for occ/dolocc
        df_data[Symbol(string(m[:modelName])*"0")] =  eval(parse(r0))
        df_data[Symbol(string(m[:modelName])*"1")] =  eval(parse(r1))
    end
end
genFixedCols(dfx,df_data,modelsDict)


function genEQ(dfx::DataFrame, m::Dict, con::String="", ranef::Symbol=:empty,level::String="") # conditional equations
    model=string(m[:modelName])
    con=strip(con)
    cbuyer_pos_p1=length(con) > 0 ? "&(df_data[:buyer_pos_p1].==1)" : "(df_data[:buyer_pos_p1].==1)"
    con = m[:Buyer_Pos_P1_is1] ? con*cbuyer_pos_p1 : con
    con = length(con) > 0 ? con*"," : con
    vout=reduce(*, map(x-> x==ranef ? "":".+df_data[$con:"*string(x)*"_"*model*"]"  ,m[:raneff]))    
    vout = ranef==:empty ? vout : vout*"+"*string(dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==model)&(dfx[:ranef].==string(ranef))&(dfx[:parameter].==level),:adj_coef][1])
    vout = ranef==:empty ? "df_data[$con:"*model*"%%]" : "df_data[$con:"*model*"%%]"*vout
    vout0 = replace(vout,"%%","0")
    vout1 = replace(vout,"%%","1")
    return vout0, vout1   # ran_mean0 = random effects are zero, so leave out
end


function genFixedMeans(dfx::DataFrame)    
    for m in mlist
        model=string(m[:modelName])
        unadj0_eq, unadj1_eq = genEQ(dfx,m,"(df_data[:group].==%%)")
        adj0_eq, adj1_eq = genEQ(dfx,m)
        if inDFX(:unadj_mean_score0_eq) dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==model)&(dfx[:parameter].=="group") ,:unadj_mean_score0_eq] = unadj0_eq end
        if inDFX(:unadj_mean_score1_eq) dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==model)&(dfx[:parameter].=="group") ,:unadj_mean_score1_eq] = unadj1_eq end
        if inDFX(:adj_mean_score0_eq) dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==model)&(dfx[:parameter].=="group") ,:adj_mean_score0_eq] = unadj0_eq end
        if inDFX(:adj_mean_score1_eq) dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==model)&(dfx[:parameter].=="group") ,:adj_mean_score1_eq] = unadj1_eq end
        unadj0 = m[:Buyer_Pos_P1_is1] ? mean(exp(eval(parse(unadj0_eq)))) : mean(exp(eval(parse(unadj0_eq))) ./ (exp(eval(parse(unadj0_eq)))+1))
        unadj1 = m[:Buyer_Pos_P1_is1] ? mean(exp(eval(parse(unadj1_eq)))) : mean(exp(eval(parse(unadj1_eq))) ./ (exp(eval(parse(unadj1_eq)))+1))
        adj0 = m[:Buyer_Pos_P1_is1] ? mean(exp(eval(parse(adj0_eq)))) : mean(exp(eval(parse(adj0_eq))) ./ (exp(eval(parse(adj0_eq)))+1))
        adj1 = m[:Buyer_Pos_P1_is1] ? mean(exp(eval(parse(adj1_eq)))) : mean(exp(eval(parse(adj1_eq))) ./ (exp(eval(parse(adj1_eq)))+1))
        dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==model)&(dfx[:parameter].=="group") ,:unadj_mean_score0] = unadj0
        dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==model)&(dfx[:parameter].=="group") ,:unadj_mean_score1] = unadj1
        dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==model)&(dfx[:parameter].=="group") ,:adj_mean_score0] = adj0
        dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==model)&(dfx[:parameter].=="group") ,:adj_mean_score1] = adj1
    end
end
genFixedMeans(dfx)


function genRandMeans(dfx::DataFrame)
    for m in mlist
        model=string(m[:modelName])
        for r in m[:raneff]
            for row in eachrow(dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].== model)&(dfx[:ranef].==string(r)),:])
                println(model," Random Effect: ",row[:ranef]," ~~ ", row[:parameter]) 
                adj0_eq, adj1_eq = genEQ(dfx,m,"",Symbol(row[:ranef]),row[:parameter])
                if inDFX(:adj_mean_score0_eq) dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==model)&(dfx[:parameter].==row[:parameter]) ,:adj_mean_score0_eq] = adj0_eq end
                if inDFX(:adj_mean_score1_eq) dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==model)&(dfx[:parameter].==row[:parameter]) ,:adj_mean_score1_eq] = adj1_eq end
                adj0 = m[:Buyer_Pos_P1_is1] ?  mean(exp(eval(parse(adj0_eq)))) :  mean(exp(eval(parse(adj0_eq))) ./ (exp(eval(parse(adj0_eq)))+1))
                adj1 = m[:Buyer_Pos_P1_is1] ?  mean(exp(eval(parse(adj1_eq)))) :  mean(exp(eval(parse(adj1_eq))) ./ (exp(eval(parse(adj1_eq)))+1)) 
                dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==model)&(dfx[:parameter].==row[:parameter]) ,:adj_mean_score0] = adj0
                dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==model)&(dfx[:parameter].==row[:parameter]) ,:adj_mean_score1] = adj1            
            end
        end        
    end
end
genRandMeans(dfx)


function genRawDataMeans(dfx::DataFrame) 
    for m in mlist   # ************************* FIXED **************************
        pre=string(m[:logvarOrig])
        post=string(m[:y_var])
        dfname="df_data"
        mname=string(m[:modelName])
        ex_Buyer_Pos_P1 = m[:Buyer_Pos_P1_is1] ? "($dfname[:buyer_pre_p1].==1)&" : ""
        for pp in ["pre","pst"]
            df_col = pp == "pre" ? pre : post
            for tc in ["cntrl","expsd"]
                ex_grp = tc == "cntrl" ? "($dfname[:group].==0) " : "($dfname[:group].==1) " 
                dfxcol=Symbol("unadj_avg_"*tc*"_hh_$pp")
                dfxcol_eq=Symbol("unadj_avg_"*tc*"_hh_"*pp*"_eq")
                ex = "mean($dfname[ $ex_Buyer_Pos_P1 $ex_grp , :$df_col])"
                println(ex)
                if inDFX(dfxcol_eq) dfx[(dfx[:parameter].=="group") & (dfx[:modelType].=="GLM") & (dfx[:model].==mname), dfxcol_eq] = ex end
                dfx[(dfx[:parameter].=="group") & (dfx[:modelType].=="GLM") & (dfx[:model].==mname), dfxcol] = eval(parse(ex))                
            end
        end
    end 
    for m in mlist  # *********************** RANDOM ************************ 
        pre=string(m[:logvarOrig])
        post=string(m[:y_var])
        dfname="df_data"
        mname=string(m[:modelName])
        ex_Buyer_Pos_P1 = m[:Buyer_Pos_P1_is1] ? "($dfname[:buyer_pre_p1].==1) &" : ""
        for r in m[:raneff]
            rs=string(r)
            for l in dfx[(dfx[:modelType].=="GLMM")&(dfx[:parameter].!="none")&(dfx[:ranef].==rs)&(dfx[:model].==mname),:parameter]
                ex_level = "($dfname[:$rs] .== \"$l\")"
                for pp in ["pre","pst"]
                    df_col = pp == "pre" ? pre : post 
                    for tc in ["cntrl","expsd"]
                        dfxcol=Symbol("unadj_avg_"*tc*"_hh_$pp")
                        break_exposed = true  # DEFAULT Hack because all Julia model data is exposed so far
                        if (tc == "cntrl") & (break_exposed)   # default to total Campaign for exposed breaks
                            fval = dfx[(dfx[:modelType].=="GLM")&(dfx[:parameter].=="group")&(dfx[:model].==mname),dfxcol][1]
                            dfx[(dfx[:modelType].=="GLMM")&(dfx[:parameter].==l)&(dfx[:ranef].==rs)&(dfx[:model].==mname),dfxcol] = fval
                        else
                            ex_grp = tc == "cntrl" ? "($dfname[:group].==0) &" : "($dfname[:group].==1) &"   
                            dfxcol_eq=Symbol("unadj_avg_"*tc*"_hh_"*pp*"_eq")
                            ex = "mean($dfname[ $ex_grp $ex_Buyer_Pos_P1 $ex_level, :$df_col])"
                            println(mname," ... ",dfxcol," ~~~ ",ex)
                            if inDFX(dfxcol_eq) dfx[(dfx[:modelType].=="GLMM")&(dfx[:parameter].==l)&(dfx[:ranef].==rs)&(dfx[:model].==mname),dfxcol_eq]=ex end
                            dfx[(dfx[:modelType].=="GLMM")&(dfx[:parameter].==l)&(dfx[:ranef].==rs)&(dfx[:model].==mname),dfxcol] = eval(parse(ex))
                        end
                    end
                end
            end
        end
    end
end
genRawDataMeans(dfx)


function getCnts(df_data::DataFrame, ranef::String="", level::String="" )
    isBreak = length(ranef) == 0 ? false : true
    ex_re = isBreak ? "& (df_data[:$ranef] .== \"$level\")" : "  "  # need to have at least 2 spaces for single col belowbelow
    ex_re_single_col = isBreak ? ex_re[2:end]*"," : ""   
    ex_m = "length(df_data[ (df_data[:buyer_pos_p1] .== 1) $ex_re ,1])"
    ex_mt="length(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pos_p1] .== 1 ) $ex_re ,1])"
    ex_n="length(df_data[ $ex_re_single_col 1])"
    ex_nt="length(df_data[ (df_data[:group] .== 1) $ex_re ,1])"
    ex_mc = "length(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pos_p1] .== 1 ) ,1])"
    ex_nc = "length(df_data[ (df_data[:group] .== 0) ,1])"
    cdict = OrderedDict()
    cdict[:M] =  eval(parse(ex_m))
    cdict[:Mt] = eval(parse(ex_mt))
    cdict[:Mc] = eval(parse(ex_mc))
    cdict[:N] =  eval(parse(ex_n))
    cdict[:Nt] = eval(parse(ex_nt))
    cdict[:Nc] = eval(parse(ex_nc))    
    if isBreak
        break_exposed = true  # DEFAULT Hack because all Julia model data is exposed so far   
        if break_exposed    #  If Exposed - default control (Nc & Mc) to total -- else count by rndfx        
            cdict[:M] = cdict[:Mt] + cdict[:Mc]   # Recalculate totals as Test + the defaulted Control
            cdict[:N] = cdict[:Nt] + cdict[:Nc]
        else # Need to recalculate MC & Nc with break conditions break
            ex_mc = "length(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pos_p1] .== 1 ) $ex_re ,1])"
            ex_nc = "length(df_data[ (df_data[:group] .== 0)  $ex_re ,1])"
            cdict[:Mc] = eval(parse(ex_mc))
            cdict[:Nc] = eval(parse(ex_nc))
        end
    end
    return cdict
end


function genCnts()
    c=getCnts(df_data)
    dfx[(dfx[:modelType].=="GLM")&(dfx[:parameter].=="group"),:M]=c[:M]
    dfx[(dfx[:modelType].=="GLM")&(dfx[:parameter].=="group"),:Mt]=c[:Mt]
    dfx[(dfx[:modelType].=="GLM")&(dfx[:parameter].=="group"),:Mc]=c[:Mc]
    dfx[(dfx[:modelType].=="GLM")&(dfx[:parameter].=="group"),:N]=c[:N]
    dfx[(dfx[:modelType].=="GLM")&(dfx[:parameter].=="group"),:Nt]=c[:Nt]
    dfx[(dfx[:modelType].=="GLM")&(dfx[:parameter].=="group"),:Nc]=c[:Nc]
    for r in unique(dfx[(dfx[:modelType].=="GLMM"),:ranef])
        for l in unique(dfx[(dfx[:modelType].=="GLMM")&(dfx[:ranef].==r),:parameter])
            c =  getCnts(df_data,r,l)
            dfx[(dfx[:modelType].=="GLMM")&(dfx[:ranef].==r)&(dfx[:parameter].==l),:M]=c[:M]
            dfx[(dfx[:modelType].=="GLMM")&(dfx[:ranef].==r)&(dfx[:parameter].==l),:Mt]=c[:Mt]
            dfx[(dfx[:modelType].=="GLMM")&(dfx[:ranef].==r)&(dfx[:parameter].==l),:Mc]=c[:Mc]
            dfx[(dfx[:modelType].=="GLMM")&(dfx[:ranef].==r)&(dfx[:parameter].==l),:N]=c[:N]
            dfx[(dfx[:modelType].=="GLMM")&(dfx[:ranef].==r)&(dfx[:parameter].==l),:Nt]=c[:Nt]
            dfx[(dfx[:modelType].=="GLMM")&(dfx[:ranef].==r)&(dfx[:parameter].==l),:Nc]=c[:Nc]
        end
    end
end
genCnts()


function ConfIntrvl()
    for row in eachrow(dfx)
        runCI = false
        if (row[:modelType] == "GLM") & (row[:parameter]=="group")
            runCI = true
            mean_score0=row[:unadj_mean_score0]  # TOTAL Confidence Intervals  # use unadj
            mean_score1=row[:unadj_mean_score1] 
            B=row[:coef]
            SE=row[:stderr] 
            Mt = row[:model]=="pen" ? row[:Nt] : row[:Mt]
            Mc = row[:model]=="pen" ? row[:Nc] : row[:Mc]
            M = row[:model]=="pen" ? row[:N] : row[:M]
            println("Total: ",mean_score0,"...",mean_score1," : ",B," ~ ",SE," ~ ",Mt," ~ ",Mc," ~ ",M)
            #ex="(($mean_score1*($Mt/$M))-($mean_score1*exp(-($B-($SE*\$z)))*($Mt/$M)))+(($mean_score0*exp(($B-($SE*\$z)))*($Mc/$M))-($mean_score0*($Mc/$M)))" 
            #println("EX ",row[:model],": ",ex)  #,"   ~~~   ", eval(parse(ex)))
        elseif row[:modelType]=="GLMM"
            runCI = true
            mean_score0=row[:adj_mean_score0] #RANDOM Confidence Intervals  # use adj
            mean_score1=row[:adj_mean_score1]
            B=row[:adj_coef]
            SE=row[:adj_stderr]
            Mt = row[:model]=="pen" ? row[:Nt] : row[:Mt]
            Mc = row[:model]=="pen" ? row[:Nc] : row[:Mc]
            M = row[:model]=="pen" ? row[:N] : row[:M]
            println("Random: ",mean_score0,"...",mean_score1," : ",row[:ranef],"_",row[:parameter]," ~ ",B," ~ ",SE," ~ ",Mt," ~ ",Mc," ~ ",M)
        end
        if runCI
            ZDict = Dict("onetail_80_pct_intrvl" => 0.84 ,"onetail_90_pct_intrvl" => 1.28, "twotail_80_pct_intrvl" => 1.28, "twotail_90_pct_intrvl" => 1.65)
            for (zscore_key, zscore) in ZDict    
                Lb_pre = (   (mean_score1*(Mt/M))    -   (mean_score1*exp(-(B-(SE*zscore)))*(Mt/M))   )  +  ## ------------ Lower Bound ---------------
                         (   (mean_score0*exp((B-(SE*zscore)))*(Mc/M))    -   (mean_score0*(Mc/M))    )
                row[Symbol(zscore_key*"_lb")] = ( Lb_pre/mean_score0 ) * 100
                Ub_pre =  (     ( mean_score1*(Mt/M) )   -   ( mean_score1*exp(-(B+(SE*zscore)))*(Mt/M))   )  +  ## ------------ Upper Bound ---------------
                          (     ( mean_score0*exp((B+(SE*zscore)))*(Mc/M))  - (mean_score0*(Mc/M) )   )
                row[Symbol(zscore_key*"_ub")] = ( Ub_pre/mean_score0 ) * 100   
                println("RAND CI $zscore_key ($zscore) LB:",row[Symbol(zscore_key*"_lb")]," ~~ UB : ", row[Symbol(zscore_key*"_ub")])
            end            
        end
    end
end
ConfIntrvl()


function removeBreaks(dfx::DataFrame)   # Shouldn't be an issue going forwward -- remove breaks that don't exist in all 3 models
    dfo=deepcopy(dfx)
    for ranef in unique(dfx[(dfx[:modelType].=="GLMM"),:ranef])
        for level in unique(dfx[(dfx[:modelType].=="GLMM")&(dfx[:ranef].==ranef),:parameter])
            cnt = length(dfx[(dfx[:modelType].=="GLMM")&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),:adj_mean_score0])
            if length(dfx[(dfx[:modelType].=="GLMM")&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),:adj_mean_score0]) != 3
                pre=length(dfo[1])
                dfo=dfo[(dfo[:modelType].*dfo[:ranef].*dfo[:parameter].!="GLMM"*ranef*level),:]
                post=length(dfo[1])
                println("Remove : $ranef $level : $cnt :::: $pre -> $post")
            end
        end
    end
    return dfo
end
dfx=removeBreaks(dfx)

# **********************************************************************************************
# ***************************************** DOLHH **********************************************
# **********************************************************************************************
function genDHHMeans(dfx::DataFrame)
    to0=dfx[(dfx[:modelType].=="GLM")&(dfx[:model].=="occ")&(dfx[:parameter].=="group"),:adj_mean_score0][1]
    ty0=dfx[(dfx[:modelType].=="GLM")&(dfx[:model].=="dolocc")&(dfx[:parameter].=="group"),:adj_mean_score0][1]
    tp0=dfx[(dfx[:modelType].=="GLM")&(dfx[:model].=="pen")&(dfx[:parameter].=="group"),:adj_mean_score0][1]
    adj_mean_score0=to0*ty0*tp0            
    to1=dfx[(dfx[:modelType].=="GLM")&(dfx[:model].=="occ")&(dfx[:parameter].=="group"),:adj_mean_score1][1]
    ty1=dfx[(dfx[:modelType].=="GLM")&(dfx[:model].=="dolocc")&(dfx[:parameter].=="group"),:adj_mean_score1][1]
    tp1=dfx[(dfx[:modelType].=="GLM")&(dfx[:model].=="pen")&(dfx[:parameter].=="group"),:adj_mean_score1][1]
    adj_mean_score1=to1*ty1*tp1
    unadj_avg_cntrl_hh_pre = mean(df_data[ (df_data[:group] .== 0), :prd_1_net_pr_pre] )
    unadj_avg_expsd_hh_pre = mean(df_data[ (df_data[:group] .== 1), :prd_1_net_pr_pre] )
    unadj_avg_cntrl_hh_pst = mean(df_data[ (df_data[:group] .== 0), :prd_1_net_pr_pos] )
    unadj_avg_expsd_hh_pst = mean(df_data[ (df_data[:group] .== 1), :prd_1_net_pr_pos] )
    pre=["total",0.0,0.0,0.0,0.0,"dolhh","ranef","DOLHH"]
    vars=[0,0,adj_mean_score0,adj_mean_score1,unadj_avg_expsd_hh_pre,unadj_avg_expsd_hh_pst,unadj_avg_cntrl_hh_pre,unadj_avg_cntrl_hh_pst]
    cis=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    cnts=convert(Array{Int64},collect(values(getCnts(df_data))))
    push!(dfx,vcat(pre,vars,cis,cnts,zeros(Int, length(names(dfx))-30)))
    for ranef in unique(dfx[(dfx[:modelType].=="GLMM"),:ranef])
        for level in unique(dfx[(dfx[:modelType].=="GLMM")&(dfx[:ranef].==ranef),:parameter])
            println(ranef," ~~ ",level)
            o=dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].=="occ")&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),:adj_mean_score0][1]
            y=dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].=="dolocc")&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),:adj_mean_score0][1]
            p=dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].=="pen")&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),:adj_mean_score0][1]
            adj_mean_cntrl_grp=o*y*p            
            o=dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].=="occ")&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),:adj_mean_score1][1]
            y=dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].=="dolocc")&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),:adj_mean_score1][1]
            p=dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].=="pen")&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),:adj_mean_score1][1]
            adj_mean_expsd_grp=o*y*p
            unadj_avg_expsd_hh_pre = mean(df_data[ (df_data[:group].==1) & (df_data[Symbol(ranef)].==level), :prd_1_net_pr_pre] )
            unadj_avg_expsd_hh_pst = mean(df_data[ (df_data[:group].==1) & (df_data[Symbol(ranef)].==level), :prd_1_net_pr_pos] )
            unadj_avg_cntrl_hh_pre = mean(df_data[ (df_data[:group].==0) & (df_data[Symbol(ranef)].==level), :prd_1_net_pr_pre] )
            unadj_avg_cntrl_hh_pst = mean(df_data[ (df_data[:group].==0) & (df_data[Symbol(ranef)].==level), :prd_1_net_pr_pos] )         
            exposed=true # temp hack - all exposed
            if exposed
                unadj_avg_cntrl_hh_pre = dfx[(dfx[:modelType].=="DOLHH")&(dfx[:parameter].=="total") ,:unadj_avg_cntrl_hh_pre][1]
                unadj_avg_cntrl_hh_pst = dfx[(dfx[:modelType].=="DOLHH")&(dfx[:parameter].=="total") ,:unadj_avg_cntrl_hh_pst][1]     
            end   
            pre=[level,0.0,0.0,0.0,0.0,"dolhh",ranef,"DOLHH"]
            vars=[0,0,adj_mean_score0,adj_mean_score1,unadj_avg_expsd_hh_pre,unadj_avg_expsd_hh_pst,unadj_avg_cntrl_hh_pre,unadj_avg_cntrl_hh_pst ]
            cis=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            cnts=convert(Array{Int64},collect(values(getCnts(df_data,ranef,level))))
            #println("LENGTH : ",length(pre),length(vars),length(cis),length(cnts))
            push!(dfx,vcat(pre,vars,cis,cnts,zeros(Int, length(names(dfx))-30)))
        end
    end  
    
end
genDHHMeans(dfx)



function collectModels(dfx::DataFrame,modelType::String,ranef::String,level::String)
    mDict = OrderedDict()
    o=dfx[(dfx[:modelType].==modelType)&(dfx[:model].=="occ")&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),:]
    y=dfx[(dfx[:modelType].==modelType)&(dfx[:model].=="dolocc")&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),:]
    p=dfx[(dfx[:modelType].==modelType)&(dfx[:model].=="pen")&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),:]
    mDict[:M]=p[:M][1]
    mDict[:Mt]=p[:Mt][1]
    mDict[:Mc]=p[:Mc][1]
    mDict[:N]=p[:N][1]
    mDict[:Mt]=p[:Nt][1]
    mDict[:Nc]=p[:Nc][1]
    mDict[:B1]=o[:adj_coef][1]   
    mDict[:B2]=y[:adj_coef][1]
    mDict[:B3]=p[:adj_coef][1] #fx[:adj_dod_effct] = ((dfx[:adj_mean_score1] .- dfx[:adj_mean_score0]) ./ dfx[:adj_mean_score0] ) *100
    mDict[:SE1]=o[:adj_stderr][1]
    mDict[:SE2]=y[:adj_stderr][1]
    mDict[:SE3]=p[:adj_stderr][1] #fx[:adj_dod_effct] = ((dfx[:adj_mean_score1] .- dfx[:adj_mean_score0]) ./ dfx[:adj_mean_score0] ) *100S
    mDict[:o_SE0]=0
    mDict[:y_SE0]=0
    mDict[:p_SE0]=0
    mDict[:o_B0]=0
    mDict[:y_B0]=0
    mDict[:p_B0]=0
    mDict[:o_mean_score0]=o[:adj_mean_score0][1]
    mDict[:o_mean_score1]=o[:adj_mean_score1][1]
    mDict[:y_mean_score0]=y[:adj_mean_score0][1]
    mDict[:y_mean_score1]=y[:adj_mean_score1][1]
    mDict[:p_mean_score0]=p[:adj_mean_score0][1]
    mDict[:p_mean_score1]=p[:adj_mean_score1][1] 
    return mDict
end


function ConfidenceIntervals(dfx::DataFrame)
    for ranef in unique(dfx[(dfx[:modelType].=="GLMM"),:ranef])
        for level in unique(dfx[(dfx[:modelType].=="GLMM")&(dfx[:ranef].==ranef),:parameter])
            mDict=collectModels(dfx,"GLMM",ranef,level)
            println("Running CI for : $ranef : $level")
            mDict[:metakey] = ranef*"~"*level        
            calcPValue_Opt(mDict)
            CIs_O(mDict)
            dfx[(dfx[:model].=="dolhh")&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),:onetail_pval] = mDict[:onetail_pval]*100
            dfx[(dfx[:model].=="dolhh")&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),:twotail_pval] = mDict[:twotail_pval]*100
            for k in keys(ZDict)
                dfx[(dfx[:model].=="dolhh")&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),Symbol(k*"_lb")] = mDict[Symbol(k*"_lb")]*100
                dfx[(dfx[:model].=="dolhh")&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),Symbol(k*"_ub")] = mDict[Symbol(k*"_ub")]*100
            end
        end
    end
end
ConfidenceIntervals(dfx)


function genReport(dfx::DataFrame)
    dfx=dfx[(dfx[:modelType].=="GLM")&(dfx[:parameter].=="group")|(dfx[:modelType].=="GLMM"),:]
    rep=[ :MODEL_DESC, :Model, :TIME_AGG_PERIOD, :START_WEEK,:END_WEEK, :dependent_variable,:CNT_EXPSD_HH, :UNADJ_AVG_EXPSD_HH_PRE,
          :UNADJ_AVG_CNTRL_HH_PRE,:UNADJ_AVG_EXPSD_HH_PST,:UNADJ_AVG_CNTRL_HH_PST,:UNADJ_DOD_EFFCT,:UNADJ_DIFF_EFFCT,
          :ADJ_MEAN_EXPSD_GRP,:ADJ_MEAN_CNTRL_GRP,:ADJ_DOD_EFFCT,:TWOTAIL_PVAL,:ONETAIL_PVAL,:ABS_DIFF,
          :DOL_DIFF,:ONETAIL_80_PCT_INTRVL_UB,:ONETAIL_80_PCT_INTRVL_LB,:ONETAIL_90_PCT_INTRVL_UB,:ONETAIL_90_PCT_INTRVL_LB,
          :TWOTAIL_80_PCT_INTRVL_UB,:TWOTAIL_80_PCT_INTRVL_LB,:TWOTAIL_90_PCT_INTRVL_UB,:TWOTAIL_90_PCT_INTRVL_LB,
          :CNT_IMPRESSIONS,:TWOTAIL_PVAL_to_Campaign,:ONETAIL_PVAL_to_Campaign,:CNT_Model_HH
        ]
    dfx[isnan(dfx[:unadj_avg_expsd_hh_pre]),:unadj_avg_expsd_hh_pre] = 0.0  # Sometime there are no records for subset
    dfx[isnan(dfx[:unadj_avg_expsd_hh_pst]),:unadj_avg_expsd_hh_pst] = 0.0  # Sometime there are no records for subset
    dfx[:adj_dod_effct] = ((dfx[:adj_mean_score1] .- dfx[:adj_mean_score0]) ./ dfx[:adj_mean_score0] ) *100
    dfx[:unadj_dod_effct] = ( (( dfx[:unadj_avg_expsd_hh_pst] .- dfx[:unadj_avg_expsd_hh_pre]) .- (dfx[:unadj_avg_cntrl_hh_pst ] .- dfx[:unadj_avg_cntrl_hh_pre]))  ./  dfx[:unadj_avg_cntrl_hh_pst] ) *100
    dfx[:unadj_diff_effct] = ((dfx[:unadj_avg_expsd_hh_pst] .- dfx[:unadj_avg_cntrl_hh_pst]) ./ dfx[:unadj_avg_cntrl_hh_pst] )*100
    dfx[:model_desc] = dfx[:ranef]*" (".*dfx[:parameter]*")"
    dfx[:abs_diff] = dfx[:adj_mean_score1] .- dfx[:adj_mean_score0]
    dfx[:dol_diff] = dfx[:adj_mean_score1] .- dfx[:adj_mean_score0]
    dfx[:cnt_model_hh] = dfx[:Nt]
    dfx[findin(dfx[:model],["occ","dolocc"]),:cnt_model_hh] = dfx[findin(dfx[:model],["occ","dolocc"]),:Mt]       
    dfx[:onetail_pval] = 1 - (dfx[:pval] ./ 2)  # sdf[:onetail_pval_raw] = (1-(sdf[:Praw] ./ 2)) * 100
    dfx[:twotail_pval] = 1 - dfx[:pval]         # sdf[:twotail_pval_raw] = (1-sdf[:Praw]) * 100
    dfx[:empty] = ""

    dfx[:cnt_expsd_hh]=0
    dfx[:cnt_impressions]=0
    dfx[:twotail_pval_to_campaign]=0.0
    dfx[:onetail_pval_to_campaign]=0.0
    dfo=dfx[[:model_desc,:empty,:empty,:empty,:empty,:model,:cnt_expsd_hh,:unadj_avg_expsd_hh_pre, :unadj_avg_expsd_hh_pst, 
             :unadj_avg_cntrl_hh_pre, :unadj_avg_cntrl_hh_pst,:unadj_dod_effct,:unadj_diff_effct,
             :adj_mean_score1,:adj_mean_score0,:adj_dod_effct,:twotail_pval,:onetail_pval,:abs_diff,:dol_diff,
             :onetail_80_pct_intrvl_ub,:onetail_80_pct_intrvl_lb,:onetail_90_pct_intrvl_ub,:onetail_90_pct_intrvl_lb,
             :twotail_80_pct_intrvl_ub,:twotail_80_pct_intrvl_lb,:twotail_90_pct_intrvl_ub,:twotail_90_pct_intrvl_lb
             ,:cnt_impressions ,:twotail_pval_to_campaign,:onetail_pval_to_campaign ,:cnt_model_hh
        ]]
    names!(dfo,rep)
end
dfz = genReport(dfx)


HERE!!!!! look at UNADJ_AVG_EXPSD_HH_PST :unadj_avg_expsd_hh_pst │ UNADJ_AVG_CNTRL_HH_PST --- they look off




# ****************************************************************************************************************************************



function hhcounts(cnts::Cnts)
    sdf=cnts.sdf 
    sdf[:hh]=0.0
    sdf[:impressions]=0.0
    sdf[:weight]=0.0
    df_hhc = readtable("hhcounts.csv",header=true); lowercase(df_hhc)
    df_hhc[:class] = map(x->lowercase(x)  ,df_hhc[:class])
    for row in eachrow(df_hhc[df_hhc[:class].!="total campaign",:])
        k=rkey(row)
        sdf[sdf[:key].==k,:hh] = row[:hh]
        sdf[sdf[:key].==k,:impressions] = row[:impressions]
    end
    #tot = sum(sdf[:hh])
    tot = by(sdf,:class, df -> sum(df[:hh]))[:x1][2]
    toti = sum(sdf[:impressions])
    sdf[sdf[:key].=="Total Campaign",:hh] = tot
    sdf[sdf[:key].=="Total Campaign",:impressions] = toti
    for row in eachrow(cnts.sdf[cnts.sdf[:key].!="Total Campaign",:])
        k=rkey(row)
        sdf[sdf[:key].==k,:weight] = row[:hh] / tot
        println(typeof(row[:hh]))
        println(row[:hh])
    end
end




function np_val(iDict::OrderedDict)
    M = get(iDict, :M, NA)
    Mt = get(iDict, :Mt, NA)
    Mc = get(iDict, :Mc, NA)
    mean_score0 = get(iDict, :adj_mean_cntrl_grp, NA)
    mean_score1 = get(iDict, :adj_mean_expsd_grp, NA)
    B = get(iDict, :B1_combo, NA)
    SE = get(iDict, :SE1_combo, NA)
    println(get(iDict,:key, NA),"~",M,"~",Mt,"~",Mc,"~",mean_score0,"~",mean_score1,"~",B,"~",SE)
    
    function f!(x, fvec)
        Lb_pre = (   (mean_score1*(Mt/M))    -   (mean_score1*exp(-(B-(SE*x[1])))*(Mt/M))   )  +
                 (   (mean_score0*exp((B-(SE*x[1])))*(Mc/M))    -   (mean_score0*(Mc/M))    )
        Lb = Lb_pre/mean_score0
        fvec[1] = Lb
    end
    r=nlsolve(f!,[0.1])
    #println(r)
    zvalue=r.zero[1]
    pvalue=2.0 * ccdf(Normal(), abs(zvalue))
    two_tail = 1-pvalue     
    one_tail = 1-(pvalue/2)
    return one_tail, two_tail
end

function applyWeights(mocc::MOcc,mdolocc::MDolOcc,mpen::MPen)
    for m in [mocc,mdolocc,mpen]
        println("Processing : ",m.modelName)
        m.reff.sdf = join(m.reff.sdf,cnts.sdf[[:key,:weight]], on = :key)
        sdf=m.reff.sdf
        #sdf[:factor]=0.0  #unadj_mean_score0
        fdf=deepcopy(m.reff.sdf)
        for c in [:unadj_avg_expsd_hh_pre,:unadj_avg_cntrl_hh_pre,:unadj_avg_expsd_hh_pst,:unadj_avg_cntrl_hh_pst,:adj_mean_expsd_grp,:adj_mean_cntrl_grp,:mean_score0,:mean_score1]
            fdf[c] = map((x,y) -> (x*y), fdf[c],fdf[:weight])
        end
        for col in [:unadj_avg_expsd_hh_pre,:unadj_avg_cntrl_hh_pre,:unadj_avg_expsd_hh_pst,:unadj_avg_cntrl_hh_pst,:adj_mean_expsd_grp,:adj_mean_cntrl_grp,:mean_score0,:mean_score1]
            tot=m.feff.sdf[1,col]
            scol=string(col)
            sdf[symbol("f_unweighted_"*scol)] = sdf[col]
            sdf[symbol("f_"*scol)] = 0.0
            lst=by(fdf[!isnan(fdf[col]),:], :class, df -> sum(df[col]))
            rename!(lst, :x1,:creative_weight)
            lst[:factor] = tot ./ lst[:creative_weight]             
            for row in eachrow(lst)
                f=row[:factor]
                sdf[sdf[:class].==row[:class], symbol("f_"*scol) ] = f
                sdf[sdf[:class].==row[:class], col ] = sdf[sdf[:class].==row[:class], col ] * f
            end
        end          
        sdf[:f_unweighted_onetail_pval] = sdf[:onetail_pval]
        sdf[:f_unweighted_twotail_pval] = sdf[:twotail_pval]
        sdf[:f_unweighted_adj_dod_effct] = sdf[:adj_dod_effct]
        
        sdf[:adj_dod_effct] = ((sdf[:adj_mean_expsd_grp] - sdf[:adj_mean_cntrl_grp])  ./ sdf[:adj_mean_cntrl_grp] ) * 100   
        for i in 1:length(sdf[1])   
            d=df2dict(sdf[i,:])
            p1, p2 = np_val(d)
            p1=p1*100
            p2=p2*100
            sdf[i,:onetail_pval] = p1
            sdf[i,:twotail_pval] = p2
        end
       # #RE-Default exposed Break Ctrl cols to total
        ##exposed
        #for col in [:unadj_avg_cntrl_hh_pre,:unadj_avg_cntrl_hh_pst,:adj_mean_cntrl_grp]
        #    totval=m.feff.sdf[1,col]
        #    sdf[sdf[:exposed].==true,col] = totval
        #end
        ##    sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mpen.feff.sdf[1,:unadj_avg_cntrl_hh_pre]
        ##    sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mpen.feff.sdf[1,:unadj_avg_cntrl_hh_pst] 
    end
end
if cfg[:counts]
    applyWeights(mocc,mdolocc,mpen)
end
 





#function genLevels()
#    cnts=OrderedDict()     #cnts["total"] = getCnts(df_data)
#    for r in unique(dfx[(dfx[:modelType].=="GLMM"),:ranef])
#        rdict= Dict()
#        for l in unique(dfx[(dfx[:modelType].=="GLMM")&(dfx[:ranef].==r),:parameter])
#            rdict[l] =  getCnts(df_data,r,l)
#        end
#        cnts[r] = rdict
#    end
#    return cnts
#end
##levels = genLevels()
##"2016 All Powercore Amazon Digital Shopper" in dfx[(dfx[:modelType].=="GLMM")&(dfx[:ranef].=="creative"),:parameter]




"""
# Function Creates Fixed regression Columns first , then adds ranef cols to fixed calc column -- much faster
#NOTES: Leave out fixed group1 - because thats test, and we account for text in the raneff's
#       reg0 = fixed_reg0 + [ranefs - referenced break/level] 
#       reg1 = fixed_reg0 + [ranefs - referenced break/level] + level adj_coef 
function genRandMeans(dfx::DataFrame)
    for m in mlist
        modelname=string(m[:modelName])
        c = m[:Buyer_Pos_P1_is1] ? "(dfname[:buyer_pos_p1].==1)," : ""
        r0,r1=genFixedEQUASION(dfx,m,"")  #no conditions = populate whole dataset col - even for occ/dolocc
        fixedcol0=Symbol(modelname*"0")
        fixedcol1=Symbol(modelname*"1")
        df_data[fixedcol0] =  eval(parse(r0))
        df_data[fixedcol1] =  eval(parse(r1))
        for r in m[:raneff]
            #@sync @parallel 
            for row in eachrow(dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].== modelname)&(dfx[:ranef].==string(r)),:])
                println(modelname," Random Effect: ",row[:ranef]," ~~ ", row[:parameter]) 
                rnd0, rnd1 =genRandEQUASION(dfx, m, c, Symbol(row[:ranef]), row[:parameter])
                #v0="df_data["replace(c,"dfname","df_data")*":"*string(fixedcol0)*"]"*rnd
                #v1="df_data["replace(c,"dfname","df_data")*":"*string(fixedcol1)*"]"*rnd
                fixed0_ex = "df_data["replace(c,"dfname","df_data")*":"*string(fixedcol0)*"]"
                v0= fixed0_ex*rnd0
                v1= fixed0_ex*rnd1
                if inDFX(:adj_mean_score0_eq) dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==modelname)&(dfx[:parameter].==row[:parameter]) ,:adj_mean_score0_eq] = v0 end
                if inDFX(:adj_mean_score1_eq) dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==modelname)&(dfx[:parameter].==row[:parameter]) ,:adj_mean_score1_eq] = v1 end                
                mr0 = m[:Buyer_Pos_P1_is1] ?  mean(exp(eval(parse(v0)))) :  mean(exp(eval(parse(v0))) ./ (exp(eval(parse(v0)))+1))
                mr1 = m[:Buyer_Pos_P1_is1] ?  mean(exp(eval(parse(v1)))) :  mean(exp(eval(parse(v1))) ./ (exp(eval(parse(v1)))+1)) 
                dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==modelname)&(dfx[:parameter].==row[:parameter]) ,:adj_mean_score0] = mr0
                dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==modelname)&(dfx[:parameter].==row[:parameter]) ,:adj_mean_score1] = mr1            
            end
        end
    end
end
genRandMeans(dfx)
#dfx[(dfx[:parameter].=="WebMD")&(dfx[:modelType].=="GLMM"),:adj_mean_score0_eq][1]
"""



"""
function genRandEQUASION(dfx::DataFrame, m::Dict, con::String, ranef::Symbol,level::String) # conditional equations
    model=string(m[:modelName])
    dfname="df_data"
    con = replace(con,"dfname",dfname)
    B=dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==model)&(dfx[:ranef].==string(ranef))&(dfx[:parameter].==level) ,:adj_coef][1]
    vout=""
    for r in setdiff(m[:raneff],[ranef])
        vout=vout*".+$dfname[$con:"*string(r)*"_"*model*"]"
    end
    vout1=vout*"+"*string(B)
    return vout, vout1   # ran_mean0 = random effects are zero, so leave out
end
#r0,r1=genRandEQUASION(dfx, modelsDict[:occ],"(dfname[:buyer_pos_p1].==1),",) # XYZ: 0/1 , dfname:dfname
"""

"""
function genFixedMeansFULL(dfx::DataFrame)    
    for m in mlist
        modelname=m[:modelName]
        modelnameS=string(modelname)
        c = m[:Buyer_Pos_P1_is1] ? "(dfname[:group].==XYZ)&(dfname[:buyer_pos_p1].==1)," : "(dfname[:group].==XYZ),"
        r0,r1=genFixedEQUASION(dfx, m,c)
        if inDFX(:unadj_mean_score0_eq) dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==modelnameS)&(dfx[:parameter].=="group") ,:unadj_mean_score0_eq] = r0 end
        if inDFX(:unadj_mean_score1_eq) dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==modelnameS)&(dfx[:parameter].=="group") ,:unadj_mean_score1_eq] = r1 end

        mr0= m[:Buyer_Pos_P1_is1] ?  mean(exp(eval(parse(r0)))) : mean(exp(eval(parse(r0))) ./ (exp(eval(parse(r0)))+1))
        mr1= m[:Buyer_Pos_P1_is1] ?  mean(exp(eval(parse(r1)))) : mean(exp(eval(parse(r1))) ./ (exp(eval(parse(r1)))+1))        
        dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==modelnameS)&(dfx[:parameter].=="group") ,:unadj_mean_score0] = mr0
        dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==modelnameS)&(dfx[:parameter].=="group") ,:unadj_mean_score1] = mr1
        
        c = m[:Buyer_Pos_P1_is1] ? "(dfname[:buyer_pos_p1].==1)," : ""
        r0,r1=genFixedEQUASION(dfx, m,c)     
        if inDFX(:adj_mean_score0_eq) dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==modelnameS)&(dfx[:parameter].=="group") ,:adj_mean_score0_eq] = r0 end
        if inDFX(:adj_mean_score1_eq) dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==modelnameS)&(dfx[:parameter].=="group") ,:adj_mean_score1_eq] = r1 end

        mr0= m[:Buyer_Pos_P1_is1] ?  mean(exp(eval(parse(r0)))) :  mean(exp(eval(parse(r0))) ./ (exp(eval(parse(r0)))+1))
        mr1= m[:Buyer_Pos_P1_is1] ?  mean(exp(eval(parse(r1)))) :  mean(exp(eval(parse(r1))) ./ (exp(eval(parse(r1)))+1))    
        dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==modelnameS)&(dfx[:parameter].=="group") ,:adj_mean_score0] = mr0
        dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==modelnameS)&(dfx[:parameter].=="group") ,:adj_mean_score1] = mr1
    end
end
#genFixedMeansFULL(dfx)  # Not used -- but good to have for other campaign types
"""







"""
function expandDFX()
    dfx[:adj_mean_score0_p] = 0.0
    dfx[:adj_mean_score1_p] = 0.0
    dfx[:unadj_mean_score0_p] = 0.0
    dfx[:unadj_mean_score1_p] = 0.0
    for row in eachrow(dfx)
        if row[:modelType] == "GLMM" || (row[:modelType] == "GLM") & (row[:parameter] == "group")
            pt = row[:model] == "pen" ?  row[:Nt]/row[:N] : row[:Mt]/row[:M]
            pc = row[:model] == "pen" ?  row[:Nc]/row[:N] : row[:Mc]/row[:M]
            row[:adj_mean_score0_p] = row[:adj_mean_score0] * pc
            row[:adj_mean_score1_p] = row[:adj_mean_score1] * pt
            row[:unadj_mean_score0_p] = row[:unadj_mean_score0] * pc 
            row[:unadj_mean_score1_p] = row[:unadj_mean_score1] * pt
        end
    end
end
expandDFX()
# TEST CIs
GLMx(model::String) = hcat(names(dfx),Array(dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==model)&(dfx[:parameter].=="group"),:])')
#GLMx("occ")
GLMMx(model::String,ranef::String,level::String) = hcat(names(dfx), Array(dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==model)&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),:])')
#GLMMx("occ","creative","all Powercore Preroll")
"""


"""

# GENERATE SEPORATE DATAFRAME FOR REGRESSION FOR EACH MODEL

dfx = readtable(campaign_fname,header=true);

function regRandGen(dfx::DataFrame, dfd::DataFrame, modelsDict::Dict,modelIN::Symbol)   # Based On campaign.csv variables
    m=modelsDict[modelIN]
    model=string(m[:modelName])
    #imodel=Symbol("i"*replace(string(model),"i",""))
    modelType="GLMM"
    dfo=DataFrame()
    for r in m[:raneff]
        sr=string(r)
        dfo[r] = m[:Buyer_Pos_P1_is1] ? dfd[ dfd[:buyer_pos_p1].==1,r] : dfd[r]
        for row in eachrow(dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model)&(dfx[:ranef].==sr) ,:])
            dfo[ (dfo[r].==row[:parameter]),r] = string(row[:coef])
            println(r," ~~ ",row[:parameter])
        end
        dfo[r] = map(x->parse(Float64,x),dfo[r])
        dfo[r] = convert(Array{Float64},dfo[r])
    end
    return dfo
end
#x = regRandGen(dfx, df_data,modelsDict,:occ)



function regFixedGen(dfx::DataFrame, dfd::DataFrame, modelsDict::Dict,modelIN::Symbol)   # Based On modelsDict
    m=modelsDict[modelIN]
    model=string(m[:modelName])
    #imodel=Symbol("i"*replace(string(model),"i",""))
    modelType="GLM"
    dfo=DataFrame()
    dfname="df_data"
    println(dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model) ,:])
    intercept=dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model)&(dfx[:parameter].=="(Intercept)") ,[:coef]][1][1]
    grp=dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model)&(dfx[:parameter].=="group") ,[:coef]][1][1]
    vout=string(intercept)*""
    dfselect=""
    dfreg=""
    results=dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model) ,[:parameter,:coef]]
    for row in eachrow( results[findin(results[:parameter],map(x->string(x),    setdiff(m[:finalvars],[:group])  )) ,[:parameter,:coef]] )
        vout=vout*"+("*dfname*"[:"*string(row[:parameter])*"].*"*string(row[:coef])*")"
        p=string(row[:parameter])
        c=string(row[:coef])
        vmeta="\"(\"*string(row[:$p])*\"*$c)+\",\n"
        dfreg=dfreg*vmeta
        dfselect=dfselect*",:"*string(row[:parameter])
    end
    dfselect=dfname*"[1,["dfselect[2:end]*"]]"
    println("\n\nfor row in eachrow("*dfselect*")")
    #println("    println("*"\"y="*string(intercept)*"+\"*"*dfreg*"\""*string(grp)*"\")")   # WITH GROUP
    println("    println("*"\"y="*string(intercept)*"+\"*"*dfreg[1:end-4]*"\")")
    println("end\n\n")
    vout1=vout*"+"*string(grp)   # WITH GROUP
    #dfd = m[:Buyer_Pos_P1_is1] ? dfd[ dfd[:buyer_pos_p1].==1, :] : dfd
    println(  "eval(parse("*vout*"))"   )
    if m[:Buyer_Pos_P1_is1]
        dfo = DataFrame(panid=dfd[:panid], buyer_pos_p1=dfd[:buyer_pos_p1], group=dfd[:group], fixed=eval(parse(vout)), fixed0=exp(eval(parse(vout))), fixed1=exp(eval(parse(vout1))))
    else
        dfo = DataFrame(panid=dfd[:panid], buyer_pos_p1=dfd[:buyer_pos_p1], group=dfd[:group], fixed=eval(parse(vout)), 
                        fixed0=exp(eval(parse(vout))) ./ (exp(eval(parse(vout)))+1) , 
                        fixed1=exp(eval(parse(vout1))) ./ (exp(eval(parse(vout1)))+1)
                       )
        #exp(df_data[:pre_pen_score0]) ./ ( exp(df_data[:pre_pen_score0]) +1)
    end
    dfo = m[:Buyer_Pos_P1_is1] ? dfo[dfo[:buyer_pos_p1].==1,:] : dfo
    dfo[:grp] = grp
    #return grp,vout, dfo   ..... #(grp,p, dftf)=regFixedGen(dfd, modelsDict,:iocc)
    return dfo
end
#dff = regFixedGen(dfx, df_data, modelsDict,:occ)
#dfd=readtable(mod_fname,header=true); 

#modelsDict[:occ][:raneff]
rocc = hcat(regFixedGen(dfx, df_data, modelsDict,:occ), regRandGen(dfx, df_data, modelsDict,:occ)  )
#t = "rocc[:tot0]="*reduce(*, [".+rocc[:"*string(c)*"]" for c in setdiff(names(rocc),[:buyer_pos_p1,:panid,:grp])])[3:end]
t = "rocc[:tot0]="*reduce(*, [".+rocc[:"*string(c)*"]" for c in vcat([:fixed],modelsDict[:occ][:raneff])])[3:end]
eval(parse(t)) 
rocc[:tot1]=rocc[:tot0].+rocc[:grp]

rdolocc = hcat(regFixedGen(dfx, df_data, modelsDict,:dolocc), regRandGen(dfx, df_data, modelsDict,:dolocc)  )
t = "rdolocc[:tot0]="*reduce(*, [".+rdolocc[:"*string(c)*"]" for c in setdiff(names(rdolocc),[:buyer_pos_p1,:panid,:grp])])[3:end]
eval(parse(t)) 
rdolocc[:tot1]=rdolocc[:tot0].+rdolocc[:grp]

rpen = hcat(regFixedGen(dfx, df_data, modelsDict,:pen), regRandGen(dfx, df_data, modelsDict,:pen)  )
t = "rpen[:tot0]="*reduce(*, [".+rpen[:"*string(c)*"]" for c in setdiff(names(rpen),[:buyer_pos_p1,:panid,:grp])])[3:end]
eval(parse(t)) 
rpen[:tot1]=rpen[:tot0].+rpen[:grp]

"""

# ==========================================================================================
# =================================== DolHH ================================================
# =========pen models = Aproximation - exact difference =================================================================================


#function insertMissingLevels(dfx::DataFrame)
#    for ranef in unique(dfx[(dfx[:modelType].=="GLMM"),:ranef])
#        for level in unique(dfx[(dfx[:modelType].=="GLMM")&(dfx[:ranef].==ranef),:parameter])
#            review(dfa::DataArray,alt::Float64) = length(dfa) > 0 ? dfa[1] : alt 
#            for m in ["occ","dolocc","pen"]
#                if length(dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==m)&(dfx[:ranef].==ranef)&(dfx[:parameter].==level),:adj_mean_score0]) == 0
#                    x= dfx[(dfx[:modelType].=="GLM")&(dfx[:parameter].=="group")&(dfx[:model].==m),:]
#                    x[:modelType]="GLMM"
#                    x[:ranef]=ranef
#                    x[:parameter]=level
#                    append!(dfx,x)
#                    println("added $m $ranef ~ $level")
#                #else
#                #    println(".....")
#                end
#            end
#        end
#    end
#    #return dfx
#end
#insertMissingLevels(dfx)




 cols=[:model_desc, :model, :time_agg_period, :start_week,:end_week, :dependent_variable,:cnt_expsd_hh, :unadj_avg_expsd_hh_pre,
          :unadj_avg_cntrl_hh_pre,:unadj_avg_expsd_hh_pst,:unadj_avg_cntrl_hh_pst,:unadj_dod_effct,:unadj_diff_effct,
          :adj_mean_expsd_grp,:adj_mean_cntrl_grp,:adj_dod_effct,:twotail_pval,:onetail_pval,:abs_diff,
          :dol_diff,:onetail_80_pct_intrvl_ub,:onetail_80_pct_intrvl_lb,:onetail_90_pct_intrvl_ub,:onetail_90_pct_intrvl_lb,
          :twotail_80_pct_intrvl_ub,:twotail_80_pct_intrvl_lb,:twotail_90_pct_intrvl_ub,:twotail_90_pct_intrvl_lb,
          :cnt_impressions,:twotail_pval_to_campaign,:onetail_pval_to_campaign,:cnt_model_hh
    ]


function extendRDF(rdf::DataFrame)
    dfo=DataFrame(MODEL_DESC=ASCIIString[], Model=ASCIIString[], TIME_AGG_PERIOD=Int64[], START_WEEK=Int64[], END_WEEK=Int64[], dependent_variable=ASCIIString[],
                  CNT_EXPSD_HH=Int64[], UNADJ_AVG_EXPSD_HH_PRE=Float64[], UNADJ_AVG_CNTRL_HH_PRE=Float64[], UNADJ_AVG_EXPSD_HH_PST=Float64[], 
                  UNADJ_AVG_CNTRL_HH_PST=Float64[], UNADJ_DOD_EFFCT=Float64[], UNADJ_DIFF_EFFCT=Float64[], ADJ_MEAN_EXPSD_GRP=Float64[],
                  ADJ_MEAN_CNTRL_GRP=Float64[], ADJ_DOD_EFFCT=Float64[], TWOTAIL_PVAL=Float64[], ONETAIL_PVAL=Float64[], ABS_DIFF=Float64[], 
                  DOL_DIFF=Float64[], ONETAIL_80_PCT_INTRVL_UB=Float64[], ONETAIL_80_PCT_INTRVL_LB=Float64[], ONETAIL_90_PCT_INTRVL_UB=Float64[],
                  ONETAIL_90_PCT_INTRVL_LB=Float64[], TWOTAIL_80_PCT_INTRVL_UB=Float64[], TWOTAIL_80_PCT_INTRVL_LB=Float64[], TWOTAIL_90_PCT_INTRVL_UB=Float64[],
                  TWOTAIL_90_PCT_INTRVL_LB=Float64[], CNT_IMPRESSIONS=Int64[], TWOTAIL_PVAL_to_Campaign=Float64[], ONETAIL_PVAL_to_Campaign=Float64[],
                  CNT_Model_HH=Float64[]
                 )

    if (!cfg[:TotalModelsOnly]) & (length(filter(x-> x.hasBreaks ,[mocc,mdolocc, mpen]))  > 0 )
            rvs=vcat( map(x-> x.reff.sdf, filter(x-> x.hasBreaks ,[mocc,mdolocc, mpen])))[[:key,:model,:twotail_pval_raw,:onetail_pval_raw]]
            rvs = vcat(rvs,mdolhh.sdf[[:key,:model,:twotail_pval_raw,:onetail_pval_raw]])
    else
        rvs = mdolhh.sdf[[:key,:model,:twotail_pval_raw,:onetail_pval_raw]]
    end
        
    for i in 1:length(rdf[1])
        push!(dfo, [
                     rdf[i,:key], 
                     NA, #Model=NA, # code
                     NA, #TIME_AGG_PERIOD=NA, 
                     NA, #START_WEEK=NA, 
                     NA, #END_WEEK=NA, 
                     rdf[i,:model],
                     NA, #CNT_EXPSD_HH=Int64[], 
                     rdf[i,:unadj_avg_expsd_hh_pre], 
                     rdf[i,:unadj_avg_cntrl_hh_pre], 
                     rdf[i,:unadj_avg_expsd_hh_pst], 
                     rdf[i,:unadj_avg_cntrl_hh_pst], 
                     NA, #UNADJ_DOD_EFFCT=Float64[], 
                     NA, #UNADJ_DIFF_EFFCT=Float64[], 
                     rdf[i,:adj_mean_expsd_grp],
                     rdf[i,:adj_mean_cntrl_grp], 
                     rdf[i,:adj_dod_effct], 
                     rdf[i,:twotail_pval], 
                     rdf[i,:onetail_pval], 
                     NA, #ABS_DIFF=Float64[], 
                     NA, #DOL_DIFF=Float64[], 
                     rdf[i,:onetail_80_pct_intrvl_ub], 
                     rdf[i,:onetail_80_pct_intrvl_lb], 
                     rdf[i,:onetail_90_pct_intrvl_ub],
                     rdf[i,:onetail_90_pct_intrvl_lb], 
                     rdf[i,:twotail_80_pct_intrvl_ub], 
                     rdf[i,:twotail_80_pct_intrvl_lb], 
                     rdf[i,:twotail_90_pct_intrvl_ub],
                     rdf[i,:twotail_90_pct_intrvl_lb], 
                     NA, #CNT_MPRESSIONS=Float64[], 
                     NA, #TWOTAIL_PVAL_to_Campaign=Float64[], 
                     NA, #ONETAIL_PVAL_to_Campaign=Float64[],
                     NA #CNT_Model_HH=Float64[]      
                   ]
             )
        x=length(dfo[1])
        dfo[x,:UNADJ_DOD_EFFCT] = ( ((dfo[x,:UNADJ_AVG_EXPSD_HH_PST] - dfo[x,:UNADJ_AVG_EXPSD_HH_PRE]) - (dfo[x,:UNADJ_AVG_CNTRL_HH_PST] - dfo[x,:UNADJ_AVG_CNTRL_HH_PRE]))  /  dfo[x,:UNADJ_AVG_CNTRL_HH_PST] ) *100
        
        dfo[x,:UNADJ_DIFF_EFFCT] = ((dfo[x,:UNADJ_AVG_EXPSD_HH_PST] - dfo[x,:UNADJ_AVG_CNTRL_HH_PST]) / dfo[x,:UNADJ_AVG_CNTRL_HH_PST] )* 100 
    
        if dfo[i,:dependent_variable] == "dolhh"
            k=dfo[x,:MODEL_DESC]
            if cfg[:counts]
                dfo[x,:CNT_EXPSD_HH] =  getk(cnts.sdf,k,:hh)   #  cnts.get(k,:hh)
                dfo[x,:CNT_IMPRESSIONS] =  getk(cnts.sdf,k,:impressions)  #cnts.get(k,:impressions)
            end
            dfo[x,:DOL_DIFF] = dfo[x,:ADJ_MEAN_EXPSD_GRP] - dfo[x,:ADJ_MEAN_CNTRL_GRP]
            #dfo[x,:CNT_Model_HH] = cnts.sdf[cnts.sdf[:key].==dfo[x,:MODEL_DESC],:M][1]
            dfo[x,:CNT_Model_HH] = cnts.sdf[cnts.sdf[:key].==dfo[x,:MODEL_DESC],:Nt][1]
            
            
        elseif dfo[i,:dependent_variable] == "pen"
            dfo[x,:CNT_Model_HH] = cnts.sdf[cnts.sdf[:key].==dfo[x,:MODEL_DESC],:Nt][1]
        else
            dfo[x,:CNT_Model_HH] = cnts.sdf[cnts.sdf[:key].==dfo[x,:MODEL_DESC],:Mt][1]
        end        
        
        if (dfo[x,:dependent_variable] in ["occ","dolocc","pen","dolhh"]) & (dfo[x,:MODEL_DESC] !== "Total Campaign")
            
            
            pv = rvs[(rvs[:key].== dfo[x,:MODEL_DESC])&(rvs[:model].== dfo[x,:dependent_variable]) ,:]
            if length(pv[1]) >=1
                dfo[i,:TWOTAIL_PVAL_to_Campaign] = pv[:twotail_pval_raw][1]
                dfo[i,:ONETAIL_PVAL_to_Campaign] = pv[:onetail_pval_raw][1]
            end
        end
         
        #format for unify
        dfo[dfo[:dependent_variable].=="occ", :dependent_variable] = "OCC" 
        dfo[dfo[:dependent_variable].=="dolocc", :dependent_variable] = "DOL/OCC" 
        dfo[dfo[:dependent_variable].=="pen", :dependent_variable] = "PEN"
        dfo[dfo[:dependent_variable].=="dolhh", :dependent_variable] = "DOL/HH"
        
    end
    dfo[:ABS_DIFF] = dfo[:ADJ_MEAN_EXPSD_GRP] - dfo[:ADJ_MEAN_CNTRL_GRP]
    
    return dfo
end
xrdf=extendRDF(genRDF(mocc,mdolocc,mpen, mdolhh,cfg) )









































MODEL_DESC	
#Model	TIME_AGG_PERIOD	START_WEEK	END_WEEK	
dependent_variable
? CNT_EXPSD_HH	----- from hhcounts file... ToDo

UDJ_AVG_EXPSD_HH_PRE	UDJ_AVG_CNTRL_HH_PRE	UDJ_AVG_EXPSD_HH_PST	UDJ_AVG_CNTRL_HH_PST	UDJ_DOD_EFFCT	UDJ_DIFF_EFFCT	
ADJ_MEAN_EXPSD_GRP	ADJ_MEAN_CNTRL_GRP	ADJ_DOD_EFFCT	TWOTAIL_PVAL	ONETAIL_PVAL	ABS_DIFF	DOL_DIFF	ONETAIL_80_PCT_INTRVL_UB	

ONETAIL_80_PCT_INTRVL_LB	ONETAIL_90_PCT_INTRVL_UB	ONETAIL_90_PCT_INTRVL_LB	TWOTAIL_80_PCT_INTRVL_UB	TWOTAIL_80_PCT_INTRVL_LB	TWOTAIL_90_PCT_INTRVL_UB	TWOTAIL_90_PCT_INTRVL_LB	

CNT_IMPRESSIONS	TWOTAIL_PVAL_to_Campaign	ONETAIL_PVAL_to_Campaign	CNT_Model_HH

dfz=dfx[ :parameter
 #:coef
 #:stderr
 #:zval
 #:pval
 :model
 :ranef
 #... but used for subset :modelType
    
 :unadj_avg_expsd_hh_pre
 :unadj_avg_cntrl_hh_pre
 :unadj_avg_expsd_hh_pst
 :unadj_avg_cntrl_hh_pst    
    
    
 #:adj_coef
 #:adj_stderr
 :onetail_pval
 :twotail_pval
 :unadj_mean_score0
 :unadj_mean_score1
 :adj_mean_score0
 :adj_mean_score1
 #:adj_mean_score0_eq
 #:adj_mean_score1_eq

 :adj_dod_effct
 :onetail_80_pct_intrvl_lb
 :onetail_80_pct_intrvl_ub
 :onetail_90_pct_intrvl_lb
 :onetail_90_pct_intrvl_ub
 :twotail_80_pct_intrvl_lb
 :twotail_80_pct_intrvl_ub
 :twotail_90_pct_intrvl_lb
 :twotail_90_pct_intrvl_ub





# **************************************************************************************************
# **************************************************************************************************
# **********************************  END DEV HERE *************************************************
# **************************************************************************************************
# **************************************************************************************************








    
#Note:
#    :adj_mean_cntrl_grp is = adj_mean_score0
#    :adj_mean_expsd_grp
    

    
    
    
"""
    #FIXED OCC
    sdf[:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:buyer_pre_p1] .== 1) & (df_data[:group] .== 0), :trps_pre_p1] )  
    sdf[:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:buyer_pre_p1] .== 1) & (df_data[:group] .== 1), :trps_pre_p1] )
    sdf[:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 0), :trps_pos_p1] )
    sdf[:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 1), :trps_pos_p1] )
    sdf[isnan(sdf[:unadj_avg_expsd_hh_pre]),:unadj_avg_expsd_hh_pre] = 0.0  #A hack to fix missing data -- 
    #FIXED DOLOCC
    sdf[:unadj_avg_cntrl_hh_pre] =  mean(df_data[ (df_data[:buyer_pre_p1] .== 1) & (df_data[:group] .== 0), :dol_per_trip_pre_p1] )
    sdf[:unadj_avg_expsd_hh_pre] =  mean(df_data[ (df_data[:buyer_pre_p1] .== 1) & (df_data[:group] .== 1), :dol_per_trip_pre_p1] )
    sdf[:unadj_avg_cntrl_hh_pst] =  mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 0), :dol_per_trip_pos_p1] )
    sdf[:unadj_avg_expsd_hh_pst] =  mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 1), :dol_per_trip_pos_p1] )
    sdf[isnan(sdf[:unadj_avg_expsd_hh_pre]),:unadj_avg_expsd_hh_pre] = 0.0  #A hack to fix missing data -- 
    #FIXED PEN
    sdf[:unadj_avg_cntrl_hh_pre ] = mean(df_data[ (df_data[:group] .== 0)  , :buyer_pre_p1] )
    sdf[:unadj_avg_expsd_hh_pre ] = mean(df_data[ (df_data[:group] .== 1)  , :buyer_pre_p1] )
    sdf[:unadj_avg_cntrl_hh_pst ] = mean(df_data[ (df_data[:group] .== 0)  , :buyer_pos_p1] )
    sdf[:unadj_avg_expsd_hh_pst ] = mean(df_data[ (df_data[:group] .== 1)  , :buyer_pos_p1] )
    sdf[isnan(sdf[:unadj_avg_expsd_hh_pre]),:unadj_avg_expsd_hh_pre] = 0.0   #A hack to fix missing data --
    
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!1 HERE HERE HERE HERE HERE HERE HERE HERE HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #RANDOM OCC
    #PEr Level
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pre_p1] .== 1) & (df_data[sranfx] .== v_level) , :trps_pre_p1] )  #(df_data[:buyer_pre_p1] .== 1)
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :trps_pos_p1] )  #(df_data[:buyer_pos_p1] .== 1)
        if exposed
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mocc.feff.sdf[1,:unadj_avg_cntrl_hh_pre]
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mocc.feff.sdf[1,:unadj_avg_cntrl_hh_pst]
        else
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pre_p1] .== 1) & (df_data[sranfx] .== v_level) , :trps_pre_p1] )  #(df_data[:buyer_pre_p1] .== 1)
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :trps_pos_p1] )  #(df_data[:buyer_pos_p1] .== 1)
        end       

    #RANDOM DOLOCC
    #PEr Level
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pre_p1] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pre_p1] )
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pos_p1] )
        
        if exposed
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mdolocc.feff.sdf[1,:unadj_avg_cntrl_hh_pre]
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mdolocc.feff.sdf[1,:unadj_avg_cntrl_hh_pst]
        else
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pre_p1] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pre_p1] ) 
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pos_p1] )
        end   
    
    #RANDOM PEN
    # Per Level
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:group] .== 1) & (df_data[Symbol(ranfx)] .== v_level) , :buyer_pre_p1] )
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:group] .== 1) & (df_data[Symbol(ranfx)] .== v_level) , :buyer_pos_p1] )
        if exposed
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mpen.feff.sdf[1,:unadj_avg_cntrl_hh_pre]
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mpen.feff.sdf[1,:unadj_avg_cntrl_hh_pst]    
        else
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[Symbol(ranfx)] .== v_level) , :buyer_pre_p1] )
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[Symbol(ranfx)] .== v_level) , :buyer_pos_p1] )
        end   
 """   


#    Note:
#    :adj_mean_cntrl_grp is = adj_mean_score0
#    :adj_mean_expsd_grp



    
    
"""
# WITH ADJ PARAMETER =tue/false --- TO-DELETE!!!!
function genFixedEQUASION(dfx::DataFrame, m::Dict, unAdjusted::Bool=false) # conditional equations 
    model=string(m[:modelName])
    modelType="GLM"
    dfname="df_data"
    #dfname = m[:Buyer_Pos_P1_is1] ? "dfd1" : "dfd"
    if m[:Buyer_Pos_P1_is1]
        adj0 = unAdjusted ?  "($dfname[:group].==0)&($dfname[:buyer_pos_p1].==1)," : "($dfname[:buyer_pos_p1].==1),"
        adj1 = unAdjusted ?  "($dfname[:group].==1)&($dfname[:buyer_pos_p1].==1)," : "($dfname[:buyer_pos_p1].==1),"
    else
        adj0 = unAdjusted ?  "($dfname[:group].==0)," : ""
        adj1 = unAdjusted ?  "($dfname[:group].==1)," : ""
    end
    println(dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model) ,:])
    intercept=dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model)&(dfx[:parameter].=="(Intercept)") ,[:coef]][1][1]
    grp=dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model)&(dfx[:parameter].=="group") ,[:coef]][1][1]
    vout0=string(intercept)*""
    vout1=string(intercept)*""
    results=dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model) ,[:parameter,:coef]]
    for row in eachrow( results[findin(results[:parameter],map(x->string(x),    setdiff(m[:finalvars],[:group])  )) ,[:parameter,:coef]] )
        vout0=vout0*"+("*dfname*"[$adj0:"*string(row[:parameter])*"].*"*string(row[:coef])*")"
        vout1=vout1*"+("*dfname*"[$adj1:"*string(row[:parameter])*"].*"*string(row[:coef])*")"
    end
    vout1=vout1*"+"*string(grp)
    return vout0, vout1
end
#r0,r1=genFixedEQUASION(dfx, modelsDict[:occ],true)
"""

"""
# same as above
function genRandEQUASION(dfx::DataFrame, m::Dict, con::String, ranef::Symbol,level::String) # conditional equations
    model=string(m[:modelName])
    dfname="df_data"
    con = replace(con,"dfname",dfname)
    #println("genRandEQUASION : ",con) #," ~~ ",m)
    r0,r1=genFixedEQUASION(dfx,m,con)
    #println("genRandEQUASION r0 : ",r0," .... ",m)
    B=dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==model)&(dfx[:ranef].==string(ranef))&(dfx[:parameter].==level) ,:adj_coef][1]
    #println("genRandEQUASION B : ",B)
    vout=""
    for r in setdiff(m[:raneff],[ranef])
        vout=vout*".+$dfname[$con:"*string(r)*"_"*model*"]"
    end
    vout=vout*"+"*string(B)
    return r0*vout , r1*vout
end
"""

"""
function genMeans(dfx::DataFrame, includeEquations::Bool=false)
    dfx[:unadj_mean_score0] = 0.0
    dfx[:unadj_mean_score1] = 0.0
    dfx[:adj_mean_score0 ] = 0.0
    dfx[:adj_mean_score1 ] = 0.0
    if includeEquations
        dfx[:unadj_mean_score0_eq] = ""
        dfx[:unadj_mean_score1_eq] = ""
        dfx[:adj_mean_score0_eq ] = ""
        dfx[:adj_mean_score1_eq ] = ""
    end
    
    for m in [modelsDict[:occ],modelsDict[:dolocc],modelsDict[:pen]]
        modelname=m[:modelName]

        if m[:Buyer_Pos_P1_is1]
            # Fixed Means OCC/DOLOCC
            r0,r1=genFixedEQUASION(dfx, modelsDict[modelname],"(dfname[:group].==XYZ)&(dfname[:buyer_pos_p1].==1),")
            if includeEquations
                dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==string(modelname))&(dfx[:parameter].=="group") ,:unadj_mean_score0_eq] = r0
                dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==string(modelname))&(dfx[:parameter].=="group") ,:unadj_mean_score1_eq] = r1
            end
            mr0=mean(exp(eval(parse(r0))))
            mr1=mean(exp(eval(parse(r1))))            
            dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==string(modelname))&(dfx[:parameter].=="group") ,:unadj_mean_score0] = mr0
            dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==string(modelname))&(dfx[:parameter].=="group") ,:unadj_mean_score1] = mr1
            
            r0,r1=genFixedEQUASION(dfx, modelsDict[modelname],"(dfname[:buyer_pos_p1].==1),") 
            if includeEquations
                dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==string(modelname))&(dfx[:parameter].=="group") ,:adj_mean_score0_eq] = r0
                dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==string(modelname))&(dfx[:parameter].=="group") ,:adj_mean_score1_eq] = r1
            end
            mr0=mean(exp(eval(parse(r0))))
            mr1=mean(exp(eval(parse(r1))))            
            dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==string(modelname))&(dfx[:parameter].=="group") ,:adj_mean_score0] = mr0
            dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==string(modelname))&(dfx[:parameter].=="group") ,:adj_mean_score1] = mr1
            
            # Random Means OCC/DOLOCC
            #for 
            
        else
            # Fixed Means PEN
            r0,r1=genFixedEQUASION(dfx, modelsDict[modelname],"(dfname[:group].==XYZ),") 
            if includeEquations
                dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==string(modelname))&(dfx[:parameter].=="group") ,:unadj_mean_score0_eq] = r0
                dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==string(modelname))&(dfx[:parameter].=="group") ,:unadj_mean_score1_eq] = r1
            end
            mr0=mean(exp(eval(parse(r0))) ./ (exp(eval(parse(r0)))+1))
            mr1=mean(exp(eval(parse(r1))) ./ (exp(eval(parse(r1)))+1))
            dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==string(modelname))&(dfx[:parameter].=="group") ,:unadj_mean_score0] = mr0
            dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==string(modelname))&(dfx[:parameter].=="group") ,:unadj_mean_score1] = mr1
            
            r0,r1=genFixedEQUASION(dfx, modelsDict[modelname],"")
            if includeEquations
                dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==string(modelname))&(dfx[:parameter].=="group") ,:adj_mean_score0_eq] = r0
                dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==string(modelname))&(dfx[:parameter].=="group") ,:adj_mean_score1_eq] = r1
            end
            mr0=mean(exp(eval(parse(r0))) ./ (exp(eval(parse(r0)))+1))
            mr1=mean(exp(eval(parse(r1))) ./ (exp(eval(parse(r1)))+1))
            dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==string(modelname))&(dfx[:parameter].=="group") ,:adj_mean_score0] = mr0
            dfx[(dfx[:modelType].=="GLM")&(dfx[:model].==string(modelname))&(dfx[:parameter].=="group") ,:adj_mean_score1] = mr1       
    
            # Random Means PEN
            
        end
        
         
    end

    #genRandEQUASION(dfx, modelsDict[:pen], "",frequency_type,"FiveSix_Exposures")
    #mean(exp(eval(parse(r1))) ./ (exp(eval(parse(r1)))+1)

    #dfx[:unadj_mean_score0] = mean(exp(eval(parse(r0))))
    #dfx[:unadj_mean_score1] = mean(exp(eval(parse(r1))))
    #dfx[:adj_mean_score0 ] = mean(exp(eval(parse(r0))))                    
    #dfx[:adj_mean_score1 ] = mean(exp(eval(parse(r1))))
    #return dfx
end

#t = genMeans(dfx)
#t[(t[:parameter].=="group")&(t[:modelType].=="GLM"),:]
genMeans(dfx,true)







# ------------ BELOW IS GOOOD ------------ 

#genMeans(dfx,true)
#  dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].=="pen")&)(dfx[:ranef].==""),:]
#dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].=="pen")&(dfx[:ranef].=="creative"),:parameter]

# Raneff -- full regression
function genRegRandMeans(dfx::DataFrame, includeEquations::Bool=false)
    for m in [modelsDict[:occ],modelsDict[:dolocc],modelsDict[:pen]]
        modelname=string(m[:modelName])
        c = m[:Buyer_Pos_P1_is1] ? "(dfname[:buyer_pos_p1].==1)," : ""
        r0,r1=genFixedEQUASION(dfx,m,c)
        for r in m[:raneff]
            #@sync @parallel 
            for row in eachrow(dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].== modelname)&(dfx[:ranef].==string(r)),:])
                println(modelname," Random Effect: ",row[:ranef]," ~~ ", row[:parameter]) 
                rnd=genRandEQUASION(dfx, m, c, Symbol(row[:ranef]), row[:parameter])
                v0=r0*rnd
                v1=r1*rnd
                if includeEquations
                    dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==modelname)&(dfx[:parameter].==row[:parameter]) ,:adj_mean_score0_eq] = v0
                    dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==modelname)&(dfx[:parameter].==row[:parameter]) ,:adj_mean_score1_eq] = v1
                end
                mr0 = m[:Buyer_Pos_P1_is1] ?  mean(exp(eval(parse(v0)))) :  mean(exp(eval(parse(v0))) ./ (exp(eval(parse(v0)))+1))
                mr1 = m[:Buyer_Pos_P1_is1] ?  mean(exp(eval(parse(v1)))) :  mean(exp(eval(parse(v1))) ./ (exp(eval(parse(v1)))+1)) 
                dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==modelname)&(dfx[:parameter].==row[:parameter]) ,:adj_mean_score0] = mr0
                dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].==modelname)&(dfx[:parameter].==row[:parameter]) ,:adj_mean_score1] = mr1            
            end
        end
    end
end
#genRegRandMeans(dfx)





#dfx[(dfx[:modelType].=="GLMM")&(dfx[:model].=="pen")&(dfx[:ranef].=="creative"),:]




# GENERATE SEPORATE DATAFRAME FOR REGRESSION FOR EACH MODEL

function regRandGen(dfx::DataFrame, dfd::DataFrame, modelsDict::Dict,modelIN::Symbol)   # Based On campaign.csv variables
    m=modelsDict[modelIN]
    model=string(m[:modelName])
    #imodel=Symbol("i"*replace(string(model),"i",""))
    modelType="GLMM"
    dfo=DataFrame()
    for r in m[:raneff]
        sr=string(r)
        dfo[r] = m[:Buyer_Pos_P1_is1] ? dfd[ dfd[:buyer_pos_p1].==1,r] : dfd[r]
        for row in eachrow(dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model)&(dfx[:ranef].==sr) ,:])
            dfo[ (dfo[r].==row[:parameter]),r] = string(row[:coef])
            println(r," ~~ ",row[:parameter])
        end
        dfo[r] = map(x->parse(Float64,x),dfo[r])
        dfo[r] = convert(Array{Float64},dfo[r])
    end
    return dfo
end
#x = regRandGen(dfx, df_data,modelsDict,:occ)



function regFixedGen(dfx::DataFrame, dfd::DataFrame, modelsDict::Dict,modelIN::Symbol)   # Based On modelsDict
    m=modelsDict[modelIN]
    model=string(m[:modelName])
    #imodel=Symbol("i"*replace(string(model),"i",""))
    modelType="GLM"
    dfo=DataFrame()
    dfname="df_data"
    println(dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model) ,:])
    intercept=dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model)&(dfx[:parameter].=="(Intercept)") ,[:coef]][1][1]
    grp=dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model)&(dfx[:parameter].=="group") ,[:coef]][1][1]
    vout=string(intercept)*""
    dfselect=""
    dfreg=""
    results=dfx[(dfx[:modelType].==modelType)&(dfx[:model].==model) ,[:parameter,:coef]]
    for row in eachrow( results[findin(results[:parameter],map(x->string(x),    setdiff(m[:finalvars],[:group])  )) ,[:parameter,:coef]] )
        vout=vout*"+("*dfname*"[:"*string(row[:parameter])*"].*"*string(row[:coef])*")"
        p=string(row[:parameter])
        c=string(row[:coef])
        vmeta="\"(\"*string(row[:$p])*\"*$c)+\",\n"
        dfreg=dfreg*vmeta
        dfselect=dfselect*",:"*string(row[:parameter])
    end
    dfselect=dfname*"[1,["dfselect[2:end]*"]]"
    println("\n\nfor row in eachrow("*dfselect*")")
    #println("    println("*"\"y="*string(intercept)*"+\"*"*dfreg*"\""*string(grp)*"\")")   # WITH GROUP
    println("    println("*"\"y="*string(intercept)*"+\"*"*dfreg[1:end-4]*"\")")
    println("end\n\n")
    vout1=vout*"+"*string(grp)   # WITH GROUP
    #dfd = m[:Buyer_Pos_P1_is1] ? dfd[ dfd[:buyer_pos_p1].==1, :] : dfd
    println(  "eval(parse("*vout*"))"   )
    if m[:Buyer_Pos_P1_is1]
        dfo = DataFrame(panid=dfd[:panid], buyer_pos_p1=dfd[:buyer_pos_p1], group=dfd[:group], fixed=eval(parse(vout)), fixed0=exp(eval(parse(vout))), fixed1=exp(eval(parse(vout1))))
    else
        dfo = DataFrame(panid=dfd[:panid], buyer_pos_p1=dfd[:buyer_pos_p1], group=dfd[:group], fixed=eval(parse(vout)), 
                        fixed0=exp(eval(parse(vout))) ./ (exp(eval(parse(vout)))+1) , 
                        fixed1=exp(eval(parse(vout1))) ./ (exp(eval(parse(vout1)))+1)
                       )
        #exp(df_data[:pre_pen_score0]) ./ ( exp(df_data[:pre_pen_score0]) +1)
    end
    dfo = m[:Buyer_Pos_P1_is1] ? dfo[dfo[:buyer_pos_p1].==1,:] : dfo
    dfo[:grp] = grp
    #return grp,vout, dfo   ..... #(grp,p, dftf)=regFixedGen(dfd, modelsDict,:iocc)
    return dfo
end
#dff = regFixedGen(dfx, df_data, modelsDict,:occ)
#dfd=readtable(mod_fname,header=true); 

#modelsDict[:occ][:raneff]
rocc = hcat(regFixedGen(dfx, df_data, modelsDict,:occ), regRandGen(dfx, df_data, modelsDict,:occ)  )
#t = "rocc[:tot0]="*reduce(*, [".+rocc[:"*string(c)*"]" for c in setdiff(names(rocc),[:buyer_pos_p1,:panid,:grp])])[3:end]
t = "rocc[:tot0]="*reduce(*, [".+rocc[:"*string(c)*"]" for c in vcat([:fixed],modelsDict[:occ][:raneff])])[3:end]
eval(parse(t)) 
rocc[:tot1]=rocc[:tot0].+rocc[:grp]

rdolocc = hcat(regFixedGen(dfx, df_data, modelsDict,:dolocc), regRandGen(dfx, df_data, modelsDict,:dolocc)  )
t = "rdolocc[:tot0]="*reduce(*, [".+rdolocc[:"*string(c)*"]" for c in setdiff(names(rdolocc),[:buyer_pos_p1,:panid,:grp])])[3:end]
eval(parse(t)) 
rdolocc[:tot1]=rdolocc[:tot0].+rdolocc[:grp]

rpen = hcat(regFixedGen(dfx, df_data, modelsDict,:pen), regRandGen(dfx, df_data, modelsDict,:pen)  )
t = "rpen[:tot0]="*reduce(*, [".+rpen[:"*string(c)*"]" for c in setdiff(names(rpen),[:buyer_pos_p1,:panid,:grp])])[3:end]
eval(parse(t)) 
rpen[:tot1]=rpen[:tot0].+rpen[:grp]


"""










"""
-------- app.cfg --------
excludedBreaks = 
excludedKeys = 
P2_Competitor = true
pvalue_lvl = 0.20
exposed_flag_var = new_exposed_flag
sigLevel = 0.2
random_demos =
random_campaigns = 
dropvars = 
scoring_vars =
TotalModelsOnly=false
"""

cfgDefaults=OrderedDict( :P2_Competitor => true
                        ,:pvalue_lvl => 0.20  #pvalue_lvl = 0.20 
                        ,:excludedBreaks => AbstractString[]    #["estimated_hh_income","hh_age","number_of_children_in_living_un","person_1_gender"]
                        ,:excludedLevels => ["none"]
                        ,:excludedKeys => AbstractString[]
                        ,:exposed_flag_var => :exposed_flag
                        ,:sigLevel => "0.2"
                        ,:random_demos => Symbol[]
                        ,:random_campaigns => Symbol[]
                        ,:dropvars => Symbol[]
                        ,:scoring_vars => Symbol[]
                        ,:TotalModelsOnly=>false
                       )


cfg = JStack.loadCFG(cfgDefaults,pwd()*"/"*"app.cfg")
cfg[:counts] = isfile(pwd()*"/hhcounts.csv") ? true : false 
if !cfg[:counts] println("\n\n WARNING: HH Count file does not exist. No weighting will be applied to the scoring.\n\n\n") end


# ------------------------

# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ SCORING -------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------

#df_data = readtable("final_data.csv",header=true); lowercase(df_data)
#df_data = readtable(mod_fname,header=true);  #lowercase(df_data)

mocc = MOcc(df_data,cfg,dfx)
mdolocc = MDolOcc(df_data,cfg,dfx)
mpen = MPen(df_data,cfg,dfx)
mdolhh=MDolHH(mocc,mdolocc,mpen)

cnts = Cnts(df_data, mocc,mdolocc,mpen,cfg[:TotalModelsOnly])

if cfg[:counts]
    hhcounts(cnts)
end

 
function aggregate!(df_data::DataFrame, focc::FOcc, cnts::Cnts)
    sdf = focc.sdf
    #sdf[:M] = cnts.get("Total Campaign")[:M][1]
    #sdf[:Mt] = cnts.get("Total Campaign")[:Mt][1]
    #sdf[:Mc] = cnts.get("Total Campaign")[:Mc][1]
    
    sdf[:M] = getk(cnts.sdf,"Total Campaign")[:M][1]   #cnts.get("Total Campaign")[:M][1]
    sdf[:Mt] = getk(cnts.sdf,"Total Campaign")[:Mt][1]   #cnts.get("Total Campaign")[:Mt][1]
    sdf[:Mc] = getk(cnts.sdf,"Total Campaign")[:Mc][1]   #cnts.get("Total Campaign")[:Mc][1]
    
    #r0,r1=genFixedEQUASION(modelsDict[:occ],true)
    r0,r1=genFixedEQUASION(dfx, modelsDict[:occ],"(dfname[:group].==XYZ)&(dfname[:buyer_pos_p1].==1),")
    sdf[:unadj_mean_score0] = mean(exp(eval(parse(r0))))
    sdf[:unadj_mean_score1] = mean(exp(eval(parse(r1))))
    #r0,r1=genFixedEQUASION(modelsDict[:occ],false)
    r0,r1=genFixedEQUASION(dfx, modelsDict[:occ],"(dfname[:buyer_pos_p1].==1),")
    sdf[:adj_mean_cntrl_grp ] = mean(exp(eval(parse(r0))))                    
    sdf[:adj_mean_expsd_grp ] = mean(exp(eval(parse(r1))))   
    
    #sdf[:mean_score0]=mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :occ_score0] )
    #sdf[:mean_score1]=mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :occ_score1] ) 
   
    
    #Notes: -- why group/no-group
    # conlogiv=cal like - logit
    #ToDo: - stderr; Why don't raneff revolve around group1; scoring(leave out group1):= mean0(0for level/break) & mean1(adjusted coef)
# none c
    #sdf[:unadj_mean_score0]=mean(df_data[(df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 0),:occ_score0])
    #sdf[:unadj_mean_score1]=mean(df_data[(df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 1),:occ_score1])

    #sdf[:adj_mean_cntrl_grp ] = mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :occ_score0] )                     
    #sdf[:adj_mean_expsd_grp ] = mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :occ_score1] )
    # -------  Occ Score
    sdf[:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:buyer_pre_p1] .== 1) & (df_data[:group] .== 0), :trps_pre_p1] )  
    sdf[:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:buyer_pre_p1] .== 1) & (df_data[:group] .== 1), :trps_pre_p1] )
    sdf[:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 0), :trps_pos_p1] )
    sdf[:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 1), :trps_pos_p1] )
    
    #A hack to fix missing data -- 
    sdf[isnan(sdf[:unadj_avg_expsd_hh_pre]),:unadj_avg_expsd_hh_pre] = 0.0
    
end
#aggregate!(df_data, mocc.feff,cnts)



function aggregate!(df_data::DataFrame, fdolocc::FDolOcc, cnts::Cnts)
    sdf = fdolocc.sdf

    sdf[:M] = getk(cnts.sdf,"Total Campaign")[:M][1]   #cnts.get("Total Campaign")[:M][1]
    sdf[:Mt] = getk(cnts.sdf,"Total Campaign")[:Mt][1]   #cnts.get("Total Campaign")[:Mt][1]
    sdf[:Mc] = getk(cnts.sdf,"Total Campaign")[:Mc][1]   #cnts.get("Total Campaign")[:Mc][1]
    
    #r0,r1=genFixedEQUASION(modelsDict[:dolocc],true)
    r0,r1=genFixedEQUASION(dfx, modelsDict[:dolocc],"(dfname[:group].==XYZ)&(dfname[:buyer_pos_p1].==1),")
    sdf[:unadj_mean_score0] = mean(exp(eval(parse(r0))))
    sdf[:unadj_mean_score1] = mean(exp(eval(parse(r1))))
    #r0,r1=genFixedEQUASION(modelsDict[:dolocc],false)
    r0,r1=genFixedEQUASION(dfx, modelsDict[:dolocc],"(dfname[:buyer_pos_p1].==1),")
    sdf[:adj_mean_cntrl_grp ] = mean(exp(eval(parse(r0))))                    
    sdf[:adj_mean_expsd_grp ] = mean(exp(eval(parse(r1))))
    #sdf[:mean_score0]=mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :dolocc_score0] )
    #sdf[:mean_score1]=mean( df_data[ df_data[:buyer_pos_p1] .== 1 ,  :dolocc_score1] )
    #sdf[:unadj_mean_score0]=mean(df_data[(df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 0),:dolocc_score0])
    #sdf[:unadj_mean_score1]=mean(df_data[(df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 1),:dolocc_score1])
    #sdf[:adj_mean_cntrl_grp ] =  mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :dolocc_score0] )             
    #sdf[:adj_mean_expsd_grp ] =  mean( df_data[ df_data[:buyer_pos_p1] .== 1 ,  :dolocc_score1] )            
    # -------  DolOcc Score
    sdf[:unadj_avg_cntrl_hh_pre] =  mean(df_data[ (df_data[:buyer_pre_p1] .== 1) & (df_data[:group] .== 0), :dol_per_trip_pre_p1] )
    sdf[:unadj_avg_expsd_hh_pre] =  mean(df_data[ (df_data[:buyer_pre_p1] .== 1) & (df_data[:group] .== 1), :dol_per_trip_pre_p1] )
    sdf[:unadj_avg_cntrl_hh_pst] =  mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 0), :dol_per_trip_pos_p1] )
    sdf[:unadj_avg_expsd_hh_pst] =  mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 1), :dol_per_trip_pos_p1] )
    
    #A hack to fix missing data -- 
    sdf[isnan(sdf[:unadj_avg_expsd_hh_pre]),:unadj_avg_expsd_hh_pre] = 0.0
    
end
#aggregate!(df_data, mdolocc.feff, cnts)


function aggregate!(df_data::DataFrame, fpen::FPen, cnts::Cnts)
    sdf = fpen.sdf

    sdf[:M] = getk(cnts.sdf,"Total Campaign")[:N][1]   #cnts.get("Total Campaign")[:M][1]
    sdf[:Mt] = getk(cnts.sdf,"Total Campaign")[:Nt][1]   #cnts.get("Total Campaign")[:Mt][1]
    sdf[:Mc] = getk(cnts.sdf,"Total Campaign")[:Nc][1]   #cnts.get("Total Campaign")[:Mc][1]
    
    #r0,r1=genFixedEQUASION(modelsDict[:pen],true)
    r0,r1=genFixedEQUASION(dfx, modelsDict[:pen],"(dfname[:group].==XYZ)&(dfname[:buyer_pos_p1].==1),")
    sdf[:unadj_mean_score0] = mean( exp(eval(parse(r0))) ./  (exp(eval(parse(r0)))+1)  )
    sdf[:unadj_mean_score1] = mean( exp(eval(parse(r1))) ./  (exp(eval(parse(r1)))+1)    )
    #r0,r1=genFixedEQUASION(modelsDict[:pen],false)
    r0,r1=genFixedEQUASION(dfx, modelsDict[:pen],"(dfname[:buyer_pos_p1].==1),")
    sdf[:adj_mean_cntrl_grp ] = mean(  exp(eval(parse(r0))) ./   (exp(eval(parse(r0)))+1)   )                    
    sdf[:adj_mean_expsd_grp ] = mean(  exp(eval(parse(r1))) ./   (exp(eval(parse(r1)))+1)  )
    

    
    
    #sdf[:mean_score0]=mean( df_data[ :pen_score0] )
    #sdf[:mean_score1]=mean( df_data[ :pen_score1] )
    #sdf[:unadj_mean_score0]=mean(df_data[(df_data[:group] .== 0),:pen_score0])
    #sdf[:unadj_mean_score1]=mean(df_data[(df_data[:group] .== 1),:pen_score1])
    #sdf[:adj_mean_cntrl_grp ] = mean( df_data[ :pen_score0] )             
    #sdf[:adj_mean_expsd_grp ] = mean( df_data[ :pen_score1] )            
    # -------  Pen Score   - buyer_pre_p1
    sdf[:unadj_avg_cntrl_hh_pre ] = mean(df_data[ (df_data[:group] .== 0)  , :buyer_pre_p1] )
    sdf[:unadj_avg_expsd_hh_pre ] = mean(df_data[ (df_data[:group] .== 1)  , :buyer_pre_p1] )
    sdf[:unadj_avg_cntrl_hh_pst ] = mean(df_data[ (df_data[:group] .== 0)  , :buyer_pos_p1] )
    sdf[:unadj_avg_expsd_hh_pst ] = mean(df_data[ (df_data[:group] .== 1)  , :buyer_pos_p1] )
    
    #A hack to fix missing data -- 
    sdf[isnan(sdf[:unadj_avg_expsd_hh_pre]),:unadj_avg_expsd_hh_pre] = 0.0
    
end
#aggregate!(df_data, mpen.feff, cnts)


function aggregateCommon!(fixedeff::FixedEffect)
    sdf = fixedeff.sdf
    #src = fixedeff.src
    sdf[:adj_dod_effct] = ((sdf[:adj_mean_expsd_grp] - sdf[:adj_mean_cntrl_grp])  ./ sdf[:adj_mean_cntrl_grp] ) * 100   
    sdf[:twotail_pval] = sdf[:P]   #twotail_pval      
    sdf[:onetail_pval] = 1 - (sdf[:twotail_pval] ./ 2)   #onetail_pval
    sdf[:twotail_pval] = 1-sdf[:P]    #redo :twotail_pval
    #For NON-SIG Models, update - adj_mean_expsd_grp (to = adj_mean_cntrl_grp) & adj_dod_effct (= 0)
    #sdf[(sdf[:twotail_pval] .< 0.80) & (sdf[:twotail_pval] .!= 0),:adj_mean_expsd_grp]      =  sdf[(sdf[:twotail_pval] .< 0.80) & (sdf[:twotail_pval] .!= 0),:adj_mean_cntrl_grp]
end
#aggregateCommon!(mocc.feff)
#aggregateCommon!(mdolocc.feff)
#aggregateCommon!(mpen.feff) 

# -------- Confidence Intervals - Total --------
function ConfidenceIntervals(feff::FixedEffect)
    sdf = feff.sdf
    for row in eachrow(sdf)
            k = row[:key]
            md = JStack.df2dict(row)    
            md[:mean_score0] = md[:unadj_mean_score0];
            md[:mean_score1] = md[:unadj_mean_score1];
            cis =  CIs(md, ZDict, 0)
            for (zscore_key,zscore) in  ZDict     
                    ubk = Symbol(zscore_key*"_ub")
                    lbk = Symbol(zscore_key*"_lb")
                    sdf[sdf[:key].==k, lbk] = md[lbk]*100
                    sdf[sdf[:key].==k, ubk] = md[ubk]*100
            end
    end
end


aggregate!(df_data, mocc.feff,cnts)
aggregate!(df_data, mpen.feff, cnts)
aggregate!(df_data, mdolocc.feff, cnts)
aggregateCommon!(mocc.feff)
aggregateCommon!(mdolocc.feff)
aggregateCommon!(mpen.feff) 
ConfidenceIntervals(mocc.feff)
ConfidenceIntervals(mdolocc.feff)
ConfidenceIntervals(mpen.feff)


# --------------------------------------------------------
# ---------- BREAKS --------------------------------------
# --------------------------------------------------------

# === Random Effects ====
if !cfg[:TotalModelsOnly]
    
function init!(cnts::Cnts, m::MModel)  
        src=m.reff.src
        # --- GROUP1 -----
        src[:Z1] = src[:B1] ./ src[:SE1]
        src[:P1] = 2.0 * ccdf(Normal(), abs(src[:Z1]))
        src[:SIG1] = map( x -> ((isnan(x))|(x > 0.2)) ?  false : true, src[:P1] ) 
        # --- GROUP0 -----
        src[:Z0] = src[:B0] ./ src[:SE0]                          
        src[:P0] = 2.0 * ccdf(Normal(), abs(src[:Z0]))                       
        src[:SIG0] = map( x -> (isnan(x)|(x > 0.2)) ?  false : true, src[:P0] )  
        
        m.reff.sdf = join(m.reff.sdf,m.reff.src[[:class,:level,:B0,:B1,:SE0,:SE1,:exposed,:P1,:key]], on = :key)
        sdf=m.reff.sdf        
        sdf[:B_fixed] = m.feff.sdf[1,:B]
        sdf[:P_fixed] = m.feff.sdf[1,:P]
        sdf[:SE_fixed] = m.feff.sdf[1,:SE]
        ##sdf[:B1_combo] = sdf[:B1] + sdf[:B_fixed] #- sdf[:B0]
        ##sdf[:SE1_combo] = sqrt(sdf[:SE_fixed].^2+sdf[:SE1].^2)
        #sdf[:B1_combo] = sdf[:B1] + sdf[:B_fixed] - sdf[:B0]
        #sdf[:SE1_combo] = sqrt(sdf[:SE_fixed].^2+sdf[:SE1].^2+sdf[:SE0].^2)
        sdf[:B1_combo] = sdf[:B1] # + sdf[:B_fixed] - sdf[:B0]
        sdf[:SE1_combo] = sdf[:SE0] #sqrt(sdf[:SE_fixed].^2+sdf[:SE1].^2+sdf[:SE0].^2)
        sdf[:P1_combo] = 2.0 * ccdf(Normal(), abs( (sdf[:B1_combo]) ./ sdf[:SE1_combo])) 
        
        if m.reff.v_model == "pen"
            for k in sdf[:key]     
                sdf[sdf[:key].==k ,:M] = getk(cnts.sdf,k)[:N][1] 
                sdf[sdf[:key].==k ,:Mt] = getk(cnts.sdf,k)[:Nt][1]
                sdf[sdf[:key].==k ,:Mc] = getk(cnts.sdf,k)[:Nc][1]
            end  
        else
            for k in sdf[:key]     
                sdf[sdf[:key].==k ,:M] = getk(cnts.sdf,k)[:M][1]
                sdf[sdf[:key].==k ,:Mt] = getk(cnts.sdf,k)[:Mt][1]
                sdf[sdf[:key].==k ,:Mc] = getk(cnts.sdf,k)[:Mc][1] 
            end   
        end 
end
init!(cnts, mocc)
init!(cnts, mdolocc)
init!(cnts, mpen)


function aggregate!(df_data::DataFrame, mocc::MOcc)
    sdf=mocc.reff.sdf
    for row = eachrow(mocc.reff.sdf)
        k = row[:key]
        ranfx = lowercase(row[:class])
        sranfx = Symbol(ranfx)
        v_level = row[:level]
        exposed = row[:exposed]  
        dest_colname0=Symbol(k*"_occ_0")
        dest_colname1=Symbol(k*"_occ_1")
        mean_score0 = mean(df_data[df_data[:buyer_pos_p1] .== 1, dest_colname0])
        mean_score1 = mean(df_data[ df_data[:buyer_pos_p1] .== 1 , dest_colname1])
        #println(mean_score1)
        sdf[sdf[:key].==k ,:mean_score0] = mean_score0
        sdf[sdf[:key].==k ,:mean_score1] = mean_score1
        sdf[sdf[:key].==k ,:adj_mean_cntrl_grp] = mean_score0
        sdf[sdf[:key].==k ,:adj_mean_expsd_grp] = mean_score1
        
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pre_p1] .== 1) & (df_data[sranfx] .== v_level) , :trps_pre_p1] )  #(df_data[:buyer_pre_p1] .== 1)
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :trps_pos_p1] )  #(df_data[:buyer_pos_p1] .== 1)
        if exposed
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mocc.feff.sdf[1,:unadj_avg_cntrl_hh_pre]
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mocc.feff.sdf[1,:unadj_avg_cntrl_hh_pst]
        else
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pre_p1] .== 1) & (df_data[sranfx] .== v_level) , :trps_pre_p1] )  #(df_data[:buyer_pre_p1] .== 1)
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :trps_pos_p1] )  #(df_data[:buyer_pos_p1] .== 1)
        end       
    end 
end
aggregate!(df_data, mocc)


function aggregate!(df_data::DataFrame, mdolocc::MDolOcc)
    sdf=mdolocc.reff.sdf
    for row = eachrow(mdolocc.reff.sdf)
        k = row[:key]
        ranfx = lowercase(row[:class])
        sranfx = Symbol(ranfx)
        v_level = row[:level]
        exposed = row[:exposed]         
        dest_colname0=Symbol(k*"_dolocc_0")
        dest_colname1=Symbol(k*"_dolocc_1")
        mean_score0 = mean(df_data[df_data[:buyer_pos_p1] .== 1, dest_colname0])
        mean_score1 = mean(df_data[ df_data[:buyer_pos_p1] .== 1 , dest_colname1])
        sdf[sdf[:key].==k ,:mean_score0] = mean_score0
        sdf[sdf[:key].==k ,:mean_score1] = mean_score1   
        sdf[sdf[:key].==k ,:adj_mean_cntrl_grp] = mean_score0
        sdf[sdf[:key].==k ,:adj_mean_expsd_grp] = mean_score1
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pre_p1] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pre_p1] )
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pos_p1] )
        
        if exposed
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mdolocc.feff.sdf[1,:unadj_avg_cntrl_hh_pre]
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mdolocc.feff.sdf[1,:unadj_avg_cntrl_hh_pst]
        else
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pre_p1] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pre_p1] ) 
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pos_p1] )
        end    
    end   
end
aggregate!(df_data, mdolocc)



function aggregate!(df_data::DataFrame,mpen::MPen)
    sdf=mpen.reff.sdf
    for row = eachrow(mpen.reff.sdf)
        k = row[:key]
        ranfx = lowercase(row[:class])
        sranfx = Symbol(ranfx)
        v_level = row[:level]
        exposed = row[:exposed]         
        dest_colname0=Symbol(k*"_pen_0")
        dest_colname1=Symbol(k*"_pen_1")    
        mean_score0 = mean(df_data[dest_colname0])
        mean_score1 = mean(df_data[dest_colname1])   #println("..... ",mean_score0," ~ ",mean_score1)
        sdf[sdf[:key].==k ,:mean_score0] = mean_score0
        sdf[sdf[:key].==k ,:mean_score1] = mean_score1 
        sdf[sdf[:key].==k ,:adj_mean_cntrl_grp] = mean_score0
        sdf[sdf[:key].==k ,:adj_mean_expsd_grp] = mean_score1
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:group] .== 1) & (df_data[Symbol(ranfx)] .== v_level) , :buyer_pre_p1] )
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:group] .== 1) & (df_data[Symbol(ranfx)] .== v_level) , :buyer_pos_p1] )
        if exposed
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mpen.feff.sdf[1,:unadj_avg_cntrl_hh_pre]
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mpen.feff.sdf[1,:unadj_avg_cntrl_hh_pst]    
        else
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[Symbol(ranfx)] .== v_level) , :buyer_pre_p1] )
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[Symbol(ranfx)] .== v_level) , :buyer_pos_p1] )
        end         
    end      
end
aggregate!(df_data, mpen)
     

    #sdf[:B1_combo] = sdf[:B1] + sdf[:B_fixed] - sdf[:B0]
    #sdf[:SE1_combo] = sqrt(sdf[:SE_fixed].^2+sdf[:SE1].^2+sdf[:SE0].^2)
    #sdf[:P1_combo] = 2.0 * ccdf(Normal(), abs( (sdf[:B1_combo]) ./ sdf[:SE1_combo])) 

function aggregateCommon!(raneff::RanEffect,cfg::OrderedDict)
    sdf = raneff.sdf
    sdf[:adj_dod_effct] = ((sdf[:adj_mean_expsd_grp] - sdf[:adj_mean_cntrl_grp])  ./ sdf[:adj_mean_cntrl_grp] ) * 100   
    sdf[:twotail_pval] = (1-sdf[:P1_combo]) * 100
    sdf[:onetail_pval] = (1-(sdf[:P1_combo] ./ 2)) * 100

    sdf[:Braw] = sdf[:B1] - sdf[:B0]
    sdf[:SEraw] = sqrt(sdf[:SE1].^2+sdf[:SE0].^2)
    sdf[:Zraw] = sdf[:Braw] ./ sdf[:SEraw]
    sdf[:Praw] = 2.0 * ccdf(Normal(), abs(sdf[:Zraw]))   
    sdf[:twotail_pval_raw] = (1-sdf[:Praw]) * 100
    sdf[:onetail_pval_raw] = (1-(sdf[:Praw] ./ 2)) * 100
    
    sigLvl=parse(Float64,get(cfg, :sigLevel, "0.2"))
    
    
end
aggregateCommon!(mocc.reff,cfg)
aggregateCommon!(mdolocc.reff,cfg)
aggregateCommon!(mpen.reff,cfg) 

#--add 3 random coefs..... combin STD_ERR
# for non-significant - set to 0... or default to total....


function np_val(iDict::OrderedDict)
    M = get(iDict, :M, NA)
    Mt = get(iDict, :Mt, NA)
    Mc = get(iDict, :Mc, NA)
    mean_score0 = get(iDict, :adj_mean_cntrl_grp, NA)
    mean_score1 = get(iDict, :adj_mean_expsd_grp, NA)
    B = get(iDict, :B1_combo, NA)
    SE = get(iDict, :SE1_combo, NA)
    #zscore = get(iDict, :zscore, NA)
    #println("NLsolve : ",iDict,"\n")
    println(get(iDict,:key, NA),"~",M,"~",Mt,"~",Mc,"~",mean_score0,"~",mean_score1,"~",B,"~",SE)
    
    function f!(x, fvec)
        #fvec[1] = exp(x[1])-45
        Lb_pre = (   (mean_score1*(Mt/M))    -   (mean_score1*exp(-(B-(SE*x[1])))*(Mt/M))   )  +
                 (   (mean_score0*exp((B-(SE*x[1])))*(Mc/M))    -   (mean_score0*(Mc/M))    )
        Lb = Lb_pre/mean_score0
        fvec[1] = Lb
        ### ------------ Lower Bound ---------------
        #Lb_pre = (   (mean_score1*(Mt/M))    -   (mean_score1*exp(-(B-(SE*zscore)))*(Mt/M))   )  +
        #         (   (mean_score0*exp((B-(SE*zscore)))*(Mc/M))    -   (mean_score0*(Mc/M))    )
        #Lb = Lb_pre/mean_score0
        ### ------------ Upper Bound ---------------
        #Ub_pre =  (     ( mean_score1*(Mt/M) )   -   ( mean_score1*exp(-(B+(SE*zscore)))*(Mt/M))   )  + 
        #          (     ( mean_score0*exp((B+(SE*zscore)))*(Mc/M))  - (mean_score0*(Mc/M) )   )
        #Ub = Ub_pre/mean_score0
        #md[Symbol(zscore_key*"_lb")] = Lb
        #md[Symbol(zscore_key*"_ub")] = Ub  
    end
    r=nlsolve(f!,[0.1])
    #println(r)
    zvalue=r.zero[1]
    pvalue=2.0 * ccdf(Normal(), abs(zvalue))
    two_tail = 1-pvalue     
    one_tail = 1-(pvalue/2)
    return one_tail, two_tail
end

function applyWeights(mocc::MOcc,mdolocc::MDolOcc,mpen::MPen)
    for m in [mocc,mdolocc,mpen]
        println("Processing : ",m.modelName)
        m.reff.sdf = join(m.reff.sdf,cnts.sdf[[:key,:weight]], on = :key)
        sdf=m.reff.sdf
        #sdf[:factor]=0.0  #unadj_mean_score0
        fdf=deepcopy(m.reff.sdf)
        for c in [:unadj_avg_expsd_hh_pre,:unadj_avg_cntrl_hh_pre,:unadj_avg_expsd_hh_pst,:unadj_avg_cntrl_hh_pst,:adj_mean_expsd_grp,:adj_mean_cntrl_grp,:mean_score0,:mean_score1]
            fdf[c] = map((x,y) -> (x*y), fdf[c],fdf[:weight])
        end
        for col in [:unadj_avg_expsd_hh_pre,:unadj_avg_cntrl_hh_pre,:unadj_avg_expsd_hh_pst,:unadj_avg_cntrl_hh_pst,:adj_mean_expsd_grp,:adj_mean_cntrl_grp,:mean_score0,:mean_score1]
            tot=m.feff.sdf[1,col]
            scol=string(col)
            sdf[Symbol("f_unweighted_"*scol)] = sdf[col]
            sdf[Symbol("f_"*scol)] = 0.0
            lst=by(fdf[!isnan(fdf[col]),:], :class, df -> sum(df[col]))
            rename!(lst, :x1,:creative_weight)
            lst[:factor] = tot ./ lst[:creative_weight]             
            for row in eachrow(lst)
                f=row[:factor]
                sdf[sdf[:class].==row[:class], Symbol("f_"*scol) ] = f
                sdf[sdf[:class].==row[:class], col ] = sdf[sdf[:class].==row[:class], col ] * f
            end
        end          
        sdf[:f_unweighted_onetail_pval] = sdf[:onetail_pval]
        sdf[:f_unweighted_twotail_pval] = sdf[:twotail_pval]
        sdf[:f_unweighted_adj_dod_effct] = sdf[:adj_dod_effct]
        
        sdf[:adj_dod_effct] = ((sdf[:adj_mean_expsd_grp] - sdf[:adj_mean_cntrl_grp])  ./ sdf[:adj_mean_cntrl_grp] ) * 100   
        for i in 1:length(sdf[1])   
            d=df2dict(sdf[i,:])
            p1, p2 = np_val(d)
            p1=p1*100
            p2=p2*100
            sdf[i,:onetail_pval] = p1
            sdf[i,:twotail_pval] = p2
        end
        
       # #RE-Default exposed Break Ctrl cols to total
        ##exposed
        #for col in [:unadj_avg_cntrl_hh_pre,:unadj_avg_cntrl_hh_pst,:adj_mean_cntrl_grp]
        #    totval=m.feff.sdf[1,col]
        #    sdf[sdf[:exposed].==true,col] = totval
        #end
        ##    sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mpen.feff.sdf[1,:unadj_avg_cntrl_hh_pre]
        ##    sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mpen.feff.sdf[1,:unadj_avg_cntrl_hh_pst] 
        
        
    end
end
if cfg[:counts]
    applyWeights(mocc,mdolocc,mpen)
end
 
# -------- Confidence Intervals - Random Effects --------
function ConfidenceIntervals(cnts::Cnts,reff::RanEffect)
    sdf = reff.sdf
    for row in eachrow(sdf)
        k = row[:key]
        md = JStack.df2dict(row)    #[:B1_combo,:SE1_combo,:mean_score0,:mean_score1,:M,:Mt,:Mc]
        md[:B] =  md[:B1_combo] # don't really need these...need to clean up
        md[:SE] =  md[:SE1_combo] # ditto
            
#        md[:M_F] = cnts.sdf[1,:M]
#        md[:Mt_F] = cnts.sdf[1,:Mt]
#        md[:Mc_F] = cnts.sdf[1,:Mc]
        #if row[:exposed]
        #    cis =  CIs(md, ZDict, 1)
        #else
        #    cis =  CIs(md, ZDict, 2)
        #end
        cis =  CIs(md, ZDict)
        for (zscore_key,zscore) in  ZDict     
                ubk = Symbol(zscore_key*"_ub")
                lbk=Symbol(zscore_key*"_lb")
                sdf[sdf[:key].==k, lbk] = md[lbk]*100
                sdf[sdf[:key].==k, ubk] = md[ubk]*100
        end
    end
end
ConfidenceIntervals(cnts, mocc.reff)
ConfidenceIntervals(cnts, mdolocc.reff)
ConfidenceIntervals(cnts, mpen.reff)

#mdolhh.sdf=mdolhh.sdf[mdolhh.sdf[:key].!="creative_groups (none)",:]
    
    
end # :TotalModelsOnly    
    


# ==========================================================================================
# =================================== DolHH ================================================
# ==========================================================================================

mdf=MDF(cnts,mocc,mdolocc,mpen)
rdf=RDF(mocc,mdolocc,mpen,mdolhh)

function aggregate!(df_data::DataFrame, mdolhh::MDolHH, smdf::DataFrame)   #mdf::MDF
    sdf=mdolhh.sdf
    xsdf = join(mdolhh.sdf, cnts.sdf[[:key,:class,:level,:exposed]], on = :key)
    for row = eachrow(xsdf)
        k = row[:key]
        #if k != "Total Campaign"
            ranfx = row[:class]
            v_level = row[:level]
            exposed = row[:exposed]
            println("getting : ",k)
            #md = df2dict(mdf.get(k))
            md = JStack.df2dict(getk(smdf,k))
            adjctl = md[:o_mean_score0] * md[:y_mean_score0] * md[:p_mean_score0]
            adjexp = md[:o_mean_score1] * md[:y_mean_score1] * md[:p_mean_score1]
            println(k," ~~ ",adjctl,"~",adjexp) #,md)            
            sdf[sdf[:key].==k,:adj_mean_cntrl_grp] = adjctl
            sdf[sdf[:key].==k,:adj_mean_expsd_grp] = adjexp       
            sdf[sdf[:key].==k,:adj_dod_effct] = ((adjexp - adjctl) ./ adjctl ) * 100
            if k == "Total Campaign"
                sdf[sdf[:key].==k,:unadj_avg_cntrl_hh_pre] =  mean(df_data[ (df_data[:group] .== 0)  , :prd_1_net_pr_pre] )
                sdf[sdf[:key].==k,:unadj_avg_expsd_hh_pre] =  mean(df_data[ (df_data[:group] .== 1)  , :prd_1_net_pr_pre] )
                sdf[sdf[:key].==k,:unadj_avg_cntrl_hh_pst] =  mean(df_data[ (df_data[:group] .== 0)  , :prd_1_net_pr_pos] )
                sdf[sdf[:key].==k,:unadj_avg_expsd_hh_pst] =  mean(df_data[ (df_data[:group] .== 1)  , :prd_1_net_pr_pos] )
            else
                sdf[sdf[:key].== k,:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:group] .== 1) & (df_data[Symbol(lowercase(ranfx))] .== v_level) , :prd_1_net_pr_pre] )
                sdf[sdf[:key].== k,:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:group] .== 1) & (df_data[Symbol(lowercase(ranfx))] .== v_level) , :prd_1_net_pr_pos] )
                if exposed
                    sdf[sdf[:key].== k,:unadj_avg_cntrl_hh_pre] = sdf[1,:unadj_avg_cntrl_hh_pre] 
                    sdf[sdf[:key].== k,:unadj_avg_cntrl_hh_pst] = sdf[1,:unadj_avg_cntrl_hh_pst]     
                else
                    sdf[sdf[:key].== k,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[Symbol(lowercase(ranfx))] .== v_level) , :prd_1_net_pr_pre] )
                    sdf[sdf[:key].== k,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[Symbol(lowercase(ranfx))] .== v_level) , :prd_1_net_pr_pos] ) 
                end            
            end
        #end
    end
end
aggregate!(df_data,mdolhh,genMDF(cnts,mocc,mdolocc,mpen))
#aggregate!(df_data,mdolhh,genMDF(cnts, mocc, mdolocc, mpen))

 


function calc_rawPvals_Opt(iDict::OrderedDict)
    dout = iDict 
    M = get(iDict, :M, NA)
    Mt = get(iDict, :Mt, NA)
    Mc = get(iDict, :Mc, NA)
    N = get(iDict, :N, NA)
    Nt = get(iDict, :Mt, NA)
    Nc = get(iDict, :Nc, NA)
    B1 = get(iDict, :B1, NA)
    B2 = get(iDict, :B2, NA)
    B3 = get(iDict, :B3, NA)
    SE1 = get(iDict, :SE1, NA)
    SE2 = get(iDict, :SE2, NA)
    SE3 = get(iDict, :SE3, NA)
    SEsq=sqrt(SE1^2+SE2^2+SE3^2)
    o_mean_score0 = get(iDict, :o_mean_score0, NA)
    o_mean_score1 = get(iDict, :o_mean_score1, NA)
    y_mean_score0 = get(iDict, :y_mean_score0, NA)
    y_mean_score1 = get(iDict, :y_mean_score1, NA)
    p_mean_score0 = get(iDict, :p_mean_score0, NA)
    p_mean_score1 =get(iDict, :p_mean_score1, NA)
    Bsum=B1+B2+B3
    dout[:Bsum] = Bsum  
    ###### PVALUE - ONE & TWO ########
    m=nothing
    m = Model(solver=NLoptSolver(algorithm=:LD_MMA, maxtime=v_ttl))
    if B1 < 0
        @variable(m, Bocc >= B1)
    else
        @variable(m, Bocc <= B1)
    end
    if B2 < 0
        @variable(m, Bdolocc >= B2)
    else
        @variable(m, Bdolocc <= B2)
    end
    if B3 < 0
        @variable(m, Bpen >= B3)
    else
        @variable(m, Bpen <= B3)
    end
    @objective(m, Max, (((Bocc+Bpen+Bdolocc)-Bsum)/SEsq ))   # z = (x-u)/se
    @NLconstraint(m, 0.00000 <= ((((p_mean_score1*(Nt/N))+(p_mean_score0*exp(Bpen)*(Nc/N)))
	                           * ((o_mean_score1*(Mt/M))+(o_mean_score0*exp(Bocc)*(Mc/M)))
	                           * ((y_mean_score1*(Mt/M))+(y_mean_score0*exp(Bdolocc)*(Mc/M)))
	                             )
	                           -(((p_mean_score1*(Nt/N)*exp(-Bpen))+(p_mean_score0*(Nc/N)))
	                           *((o_mean_score1*(Mt/M)*exp(-Bocc))+(o_mean_score0*(Mc/M)))
	                           *((y_mean_score1*(Mt/M)*exp(-Bdolocc))+(y_mean_score0*(Mc/M)))
	                             )
	                          ) 
	               <= 0.00001
                    )
    #print(m)
    status = solve(m)
    zvalue=getobjectivevalue(m)
    pvalue=2.0 * ccdf(Normal(), abs(zvalue))
    two_tail = 1-pvalue     
    one_tail = 1-(pvalue/2)
    dout[:onetail_pval] = one_tail
    dout[:twotail_pval] = two_tail
    println("z-value: ", string(zvalue)," --> p-value: ",string(two_tail))
    return dout           
end

function rawPvals(mdolhh::MDolHH,x::DataFrame)
    sdf=mdolhh.sdf       
    sdf[:twotail_pval_raw] = 0.0
    sdf[:onetail_pval_raw]= 0.0
    for row = eachrow(x)
        k = row[:key]
        if k != "Total Campaign"
            md = JStack.df2dict(x[x[:key].==k,:])
            md[:metakey] = k
            println("Raw PValue : ",k)
            if :B1_orig in keys(md)  #!!!!!!!!!
                md[:B1] = md[:B1_orig]
                md[:SE1] = md[:SE1_orig]
                md[:B2] = md[:B2_orig]
                md[:SE2] = md[:SE2_orig]
                md[:B3] = md[:B3_orig]
                md[:SE3] = md[:SE3_orig]
            println(md)
                calc_rawPvals_Opt(md)
                sdf[sdf[:key].==k, :onetail_pval_raw]=md[:onetail_pval]*100
                sdf[sdf[:key].==k, :twotail_pval_raw]=md[:twotail_pval]*100
            end
        end
    end
end
#rawPvals(mdolhh,genRndMDF(cnts, mocc, mdolocc, mpen, false))
rawPvals(mdolhh,genMDF(cnts, mocc, mdolocc, mpen))
#mdolhh.sdf[[:key,:twotail_pval,:onetail_pval,:twotail_pval_raw,:onetail_pval_raw]]


function ConfidenceIntervals(mdolhh::MDolHH,x::DataFrame)
    sdf=mdolhh.sdf       
    for row = eachrow(x)
        k = row[:key]
        md = JStack.df2dict(x[x[:key].==k,:])
        md[:metakey] = k
        println("getting : ",k)
        calcPValue_Opt(md)
        CIs_O(md)
        for zk in [ :onetail_pval, :twotail_pval, :onetail_80_pct_intrvl_lb,:onetail_80_pct_intrvl_ub, :onetail_90_pct_intrvl_lb, :onetail_90_pct_intrvl_ub
                    ,:twotail_80_pct_intrvl_lb, :twotail_80_pct_intrvl_ub, :twotail_90_pct_intrvl_lb, :twotail_90_pct_intrvl_ub    
                  ]
            sdf[ sdf[:key].==k, zk]= md[zk]*100
        end
    end
end
#ConfidenceIntervals(mdolhh,genRndMDF(cnts, mocc, mdolocc, mpen, false))
#ConfidenceIntervals(mdolhh,genFixedMDF(cnts, mocc, mdolocc, mpen))
ConfidenceIntervals(mdolhh,genMDF(cnts, mocc, mdolocc, mpen))


#function vcatreff(mocc, mdolocc, mpen)
#    #dfo=DataFrame()
#    for m in [mocc,mdolocc, mpen]
#        if m.hasBreaks
#            
#    end
#end
#    function rmap(m::MModel)
#        return m.hasBreaks
#    end
#        rvs=vcat(map(x->if x.hasBreaks rmap(x) end,[mocc,mdolocc, mpen]))



function extendRDF(rdf::DataFrame)
    dfo=DataFrame(MODEL_DESC=String[], Model=String[], TIME_AGG_PERIOD=Int64[], START_WEEK=Int64[], END_WEEK=Int64[], dependent_variable=String[],
                  CNT_EXPSD_HH=Int64[], UNADJ_AVG_EXPSD_HH_PRE=Float64[], UNADJ_AVG_CNTRL_HH_PRE=Float64[], UNADJ_AVG_EXPSD_HH_PST=Float64[], 
                  UNADJ_AVG_CNTRL_HH_PST=Float64[], UNADJ_DOD_EFFCT=Float64[], UNADJ_DIFF_EFFCT=Float64[], ADJ_MEAN_EXPSD_GRP=Float64[],
                  ADJ_MEAN_CNTRL_GRP=Float64[], ADJ_DOD_EFFCT=Float64[], TWOTAIL_PVAL=Float64[], ONETAIL_PVAL=Float64[], ABS_DIFF=Float64[], 
                  DOL_DIFF=Float64[], ONETAIL_80_PCT_INTRVL_UB=Float64[], ONETAIL_80_PCT_INTRVL_LB=Float64[], ONETAIL_90_PCT_INTRVL_UB=Float64[],
                  ONETAIL_90_PCT_INTRVL_LB=Float64[], TWOTAIL_80_PCT_INTRVL_UB=Float64[], TWOTAIL_80_PCT_INTRVL_LB=Float64[], TWOTAIL_90_PCT_INTRVL_UB=Float64[],
                  TWOTAIL_90_PCT_INTRVL_LB=Float64[], CNT_IMPRESSIONS=Int64[], TWOTAIL_PVAL_to_Campaign=Float64[], ONETAIL_PVAL_to_Campaign=Float64[],
                  CNT_Model_HH=Float64[]
                 )

  ###NBNBNB : this needs to come out - only to hack total only        
  #          rvs = vcat(mocc.reff.sdf[[:key,:model,:twotail_pval_raw,:onetail_pval_raw]],                                    
#               mdolocc.reff.sdf[[:key,:model,:twotail_pval_raw,:onetail_pval_raw]],
#               mpen.reff.sdf[[:key,:model,:twotail_pval_raw,:onetail_pval_raw]],
##               mdolhh.sdf[[:key,:model,:twotail_pval_raw,:onetail_pval_raw]]
#              )
    if (!cfg[:TotalModelsOnly]) & (length(filter(x-> x.hasBreaks ,[mocc,mdolocc, mpen]))  > 0 )
            rvs=vcat( map(x-> x.reff.sdf, filter(x-> x.hasBreaks ,[mocc,mdolocc, mpen])))[[:key,:model,:twotail_pval_raw,:onetail_pval_raw]]
            rvs = vcat(rvs,mdolhh.sdf[[:key,:model,:twotail_pval_raw,:onetail_pval_raw]])
    else
        rvs = mdolhh.sdf[[:key,:model,:twotail_pval_raw,:onetail_pval_raw]]
    end
        
    for i in 1:length(rdf[1])
        push!(dfo, [
                     rdf[i,:key], 
                     NA, #Model=NA, # code
                     NA, #TIME_AGG_PERIOD=NA, 
                     NA, #START_WEEK=NA, 
                     NA, #END_WEEK=NA, 
                     rdf[i,:model],
                     NA, #CNT_EXPSD_HH=Int64[], 
                     rdf[i,:unadj_avg_expsd_hh_pre], 
                     rdf[i,:unadj_avg_cntrl_hh_pre], 
                     rdf[i,:unadj_avg_expsd_hh_pst], 
                     rdf[i,:unadj_avg_cntrl_hh_pst], 
                     NA, #UNADJ_DOD_EFFCT=Float64[], 
                     NA, #UNADJ_DIFF_EFFCT=Float64[], 
                     rdf[i,:adj_mean_expsd_grp],
                     rdf[i,:adj_mean_cntrl_grp], 
                     rdf[i,:adj_dod_effct], 
                     rdf[i,:twotail_pval], 
                     rdf[i,:onetail_pval], 
                     NA, #ABS_DIFF=Float64[], 
                     NA, #DOL_DIFF=Float64[], 
                     rdf[i,:onetail_80_pct_intrvl_ub], 
                     rdf[i,:onetail_80_pct_intrvl_lb], 
                     rdf[i,:onetail_90_pct_intrvl_ub],
                     rdf[i,:onetail_90_pct_intrvl_lb], 
                     rdf[i,:twotail_80_pct_intrvl_ub], 
                     rdf[i,:twotail_80_pct_intrvl_lb], 
                     rdf[i,:twotail_90_pct_intrvl_ub],
                     rdf[i,:twotail_90_pct_intrvl_lb], 
                     NA, #CNT_MPRESSIONS=Float64[], 
                     NA, #TWOTAIL_PVAL_to_Campaign=Float64[], 
                     NA, #ONETAIL_PVAL_to_Campaign=Float64[],
                     NA #CNT_Model_HH=Float64[]      
                   ]
             )
        x=length(dfo[1])
        dfo[x,:UNADJ_DOD_EFFCT] = ( ((dfo[x,:UNADJ_AVG_EXPSD_HH_PST] - dfo[x,:UNADJ_AVG_EXPSD_HH_PRE]) - (dfo[x,:UNADJ_AVG_CNTRL_HH_PST] - dfo[x,:UNADJ_AVG_CNTRL_HH_PRE]))  /  dfo[x,:UNADJ_AVG_CNTRL_HH_PST] ) *100
        
        dfo[x,:UNADJ_DIFF_EFFCT] = ((dfo[x,:UNADJ_AVG_EXPSD_HH_PST] - dfo[x,:UNADJ_AVG_CNTRL_HH_PST]) / dfo[x,:UNADJ_AVG_CNTRL_HH_PST] )* 100 
    
        if dfo[i,:dependent_variable] == "dolhh"
            k=dfo[x,:MODEL_DESC]
            if cfg[:counts]
                dfo[x,:CNT_EXPSD_HH] =  getk(cnts.sdf,k,:hh)   #  cnts.get(k,:hh)
                dfo[x,:CNT_IMPRESSIONS] =  getk(cnts.sdf,k,:impressions)  #cnts.get(k,:impressions)
            end
            dfo[x,:DOL_DIFF] = dfo[x,:ADJ_MEAN_EXPSD_GRP] - dfo[x,:ADJ_MEAN_CNTRL_GRP]
            dfo[x,:CNT_Model_HH] = cnts.sdf[cnts.sdf[:key].==dfo[x,:MODEL_DESC],:M][1]
            
            
        elseif dfo[i,:dependent_variable] == "pen"
            dfo[x,:CNT_Model_HH] = cnts.sdf[cnts.sdf[:key].==dfo[x,:MODEL_DESC],:Nt][1]
        else
            dfo[x,:CNT_Model_HH] = cnts.sdf[cnts.sdf[:key].==dfo[x,:MODEL_DESC],:Mt][1]
        end        
        
        if (dfo[x,:dependent_variable] in ["occ","dolocc","pen","dolhh"]) & (dfo[x,:MODEL_DESC] !== "Total Campaign")
            
            
            pv = rvs[(rvs[:key].== dfo[x,:MODEL_DESC])&(rvs[:model].== dfo[x,:dependent_variable]) ,:]
            if length(pv[1]) >=1
                dfo[i,:TWOTAIL_PVAL_to_Campaign] = pv[:twotail_pval_raw][1]
                dfo[i,:ONETAIL_PVAL_to_Campaign] = pv[:onetail_pval_raw][1]
            end
        end
         
        #format for unify
        dfo[dfo[:dependent_variable].=="occ", :dependent_variable] = "OCC" 
        dfo[dfo[:dependent_variable].=="dolocc", :dependent_variable] = "DOL/OCC" 
        dfo[dfo[:dependent_variable].=="pen", :dependent_variable] = "PEN"
        dfo[dfo[:dependent_variable].=="dolhh", :dependent_variable] = "DOL/HH"
        
    end
    dfo[:ABS_DIFF] = dfo[:ADJ_MEAN_EXPSD_GRP] - dfo[:ADJ_MEAN_CNTRL_GRP]
    
    return dfo
end
xrdf=extendRDF(genRDF(mocc,mdolocc,mpen, mdolhh,cfg) )

#xrdf[[:MODEL_DESC,:dependent_variable,:ADJ_MEAN_EXPSD_GRP,:ADJ_MEAN_CNTRL_GRP]]
#x=genMDF(cnts,mocc,mdolocc,mpen)
#x[[:key,:o_mean_score0,:y_mean_score0,:p_mean_score0,:o_mean_score1,:y_mean_score1,:p_mean_score1]]
#write2disk(xrdf,"/home/rmadmin/g/StatStack/src/xrdf.csv")
write2disk(xrdf,root*"/scored.csv")









