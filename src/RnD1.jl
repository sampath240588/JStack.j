                
#addprocs([("iriadmin@10.63.36.22", 1), ("iriadmin@10.63.36.23", 1)])
#addprocs(2)
#using DataStructures, DataArrays , DataFrames, StatsFuns, GLM , Distributions, MixedModels, StatsBase, JSON, StatLib
#@everywhere using DataStructures, DataArrays , DataFrames, StatsFuns, GLM , Distributions, MixedModels, StatsBase, JSON, StatLib


#root="/mnt/resource/analytics/models/Jenny-o"
#root="/mnt/resource/analytics/models/NatChoice"
#root="/mnt/resource/analytics/models/rev"
#root="/mnt/resource/analytics/models/ALL#10"
#root="/mnt/resource/analytics/models/ALL#11"
#root="/mnt/resource/analytics/models/CDW#6"
#root="/mnt/resource/analytics/models/HormelChili#8"
#root="/mnt/resource/analytics/models/Rev#8"                 
#root="/mnt/resource/analytics/models/Jenny-o"
using DataStructures, DataArrays , DataFrames, StatsFuns, GLM , Distributions, MixedModels, StatsBase, JSON, StatLib
root="/mnt/resource/analytics/models/ALL#10"
jmod_fname = root*"/dfd_model.json"
mod_fname = root*"/dfd_model.csv"         
modelsDict = readModels(jmod_fname) 
dfd = readtable(mod_fname,header=true);             
poolit!(dfd,modelsDict[:factors])
for m in [modelsDict[:iocc], modelsDict[:idolocc], modelsDict[:ipen]]
    for r in m[:raneff]
        dfd[r] = relevel(dfd[r], "none")
    end
end
m=modelsDict[:iocc]

#GLM
f=genF(m[:y_var],m[:finalvars])
g= m[:Buyer_Pos_P1_is1] ? glm(f, dfd[(dfd[:buyer_pos_p1].==1),:] , m[:dist], m[:lnk]) : glm(f, dfd , m[:dist], m[:lnk])
#GLMM
f=genF(m[:y_var],setdiff(m[:finalvars],[:group]),[   r=m[:raneff][2]   ])
gmm = m[:Buyer_Pos_P1_is1] ? fit!(glmm(f, dfd[(dfd[:buyer_pos_p1].==1),:] , m[:dist], m[:lnk]),false,0)   : fit!(glmm(f, dfd, m[:dist], m[:lnk]),false,0)
raneffect(gmm)



ex_re = isBreak ? "& (df_data[:$ranef] .== \"$level\")" : "  "
cdict[:M] = cdict[:Mt] + cdict[:Mc]   # Recalculate totals as Test + the defaulted Control
cdict[:N] = cdict[:Nt] + cdict[:Nc]



#
cond="(dfd[:buyer_pos_p1].==1),"
r=string(coef(g)[1])*reduce(*,map((c,a)-> "+"*string(c)*"*dfd[$cond:"string(a)*"]" , coef(g)[2:end], setdiff(f.rhs.args,[:+,1])))
y=mean(exp(eval(parse(r))))                            # 1.238803537583254
raw = mean(dfd[(dfd[:buyer_pos_p1].==1),:trps_pos_p1]) # 1.2388035126234906
rawEQpred = round(mean(exp(eval(parse(r)))),6) == round(mean(dfd[(dfd[:buyer_pos_p1].==1),:trps_pos_p1]),6)


#MEAN SCORES
cond="(dfd[:buyer_pos_p1].==1)&(dfd[:group].==%%),"
r=string(coef(g)[1])*reduce(*,map((c,a)-> "+"*string(c)*"*dfd[$cond:"string(a)*"]" , coef(g)[2:end], setdiff(f.rhs.args,[:+,1])))
o_mean_score0=mean(exp(eval(parse( replace(r,"%%","0") ))))
o_mean_score1=mean(exp(eval(parse( replace(r,"%%","1") ))))


ex_re = "   "
isBreak=false
ex_re_single_col = isBreak ? ex_re[2:end]*"," : ""   

mt=eval(parse("length(dfd[ (dfd[:group] .== 1) & (dfd[:buyer_pos_p1] .== 1 ) $ex_re ,1])" ))
n=eval(parse("length(dfd[ $ex_re_single_col 1])"))
nt=eval(parse("length(dfd[ (dfd[:group] .== 1) $ex_re ,1])"))
mc = eval(parse("length(dfd[ (dfd[:group] .== 0) & (dfd[:buyer_pos_p1] .== 1 ) ,1])"))
nc = eval(parse("length(dfd[ (dfd[:group] .== 0) ,1])"))

ctrl=o_mean_score0*(nc/n)
tst=o_mean_score1*(nt/n)

tst+ctrl == y  # it should be close



# ------------------------------------------------------------

NOTE: for fixed you're already got mean conditional on group 
    SOOOO:  exp ::: mean(y estimate for only group X) *    B+-Z*stderr.....
    
b4*C8 = control proportions
B5*C7 = test proportions
                
  LB =   (B5*C7+    B4*C8*EXP( (B1-0.85*B2))  )   -
         (B4*C8+    B5*C7*EXP(-(B1-0.85*B2))  )
  UB = 
         (B5*C7+    B4*C8*EXP( (B1+0.85*B2))  )   - 
         (B4*C8+    B5*C7*EXP(-(B1+0.85*B2))  )


  LB =   (test+    ctrl*EXP( (B1-0.85*B2))  )   -
         (ctrl+    test*EXP(-(B1-0.85*B2))  )
  UB = 
         (test+    ctrl*EXP( (B1+0.85*B2))  )   - 
         (ctrl+    test*EXP(-(B1+0.85*B2))  )



LB
                 
                Lb_pre = (   (mean_score1*(Mt/M))    -   (mean_score1*exp(-(B-(SE*zscore)))*(Mt/M))   )  +  ## ------------ Lower Bound ---------------
                         (   (mean_score0*exp((B-(SE*zscore)))*(Mc/M))    -   (mean_score0*(Mc/M))    )
                row[Symbol(zscore_key*"_lb")] = ( Lb_pre/mean_score0 ) * 100
    
UB             Ub_pre =  (     ( mean_score1*(Mt/M) )   -   ( mean_score1*exp(-(B+(SE*zscore)))*(Mt/M))   )  +  ## ------------ Upper Bound ---------------
                          (     ( mean_score0*exp((B+(SE*zscore)))*(Mc/M))  - (mean_score0*(Mc/M) )   )
                row[Symbol(zscore_key*"_ub")] = ( Ub_pre/mean_score0 ) * 100          
             
 

