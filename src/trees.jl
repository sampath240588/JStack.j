""" 
# DataSet
drop table if exists jennie.jennie6_breaks_RF;
CREATE EXTERNAL TABLE IF NOT EXISTS jennie.jennie6_breaks_RF
  ( household_id string,
    iri_week string,
    date1 string,
    creative_id int,
    placement_id int,
    gross_imps bigint,
    val_imps bigint,
    placement_nm string,
    creative_nm string,
    publisher string
   )
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY ‘\n’ STORED AS TEXTFILE LOCATION '/mapr/mapr04p/analytics0001/analytic_users/Models/trees/tables/';



# show create table jennie.jennie6_breaks_RF;

Insert Overwrite Table jennie.jennie6_breaks_RF
select distinct
b.household_id ,
a.iri_week ,
a.date1 ,
a.creative_id ,
a.placement_id ,
a.gross_imps ,
a.val_imps ,
c.placement_nm,
d.creative_nm, 
c.publisher
from daily.daily_unioned a
left join jennie.placements6 c on a.placement_id = c.id
left join jennie.creatives6 d on a.creative_id = d.id
inner JOIN WH_SUPPLMENTAL.HOUSEHOLD_DEPENDENT_ID_MAP b on a.lr_id = b.dependent_id 
where b.dependent_type = 'LVRAMP_ID' 
      and b.household_type = 'EXPERIAN' 
      and b.retailer = 'COMSCORE' 
      and b.data_supplier = 'EXPERIAN' 
      and b.current_state = 'MATCHED'
      and a.clientid in('21884504') 
      and iri_week < 1934
order by household_id, date1;

=============================================================================

select distinct
b.household_id ,
a.iri_week ,
a.date1 ,
a.creative_id ,
a.placement_id ,
a.gross_imps ,
a.val_imps 
from daily.daily_unioned a,
WH_SUPPLMENTAL.HOUSEHOLD_DEPENDENT_ID_MAP b 
where a.lr_id = b.dependent_id 
      and b.dependent_type = 'LVRAMP_ID' 
      and b.household_type = 'EXPERIAN' 
      and b.retailer = 'COMSCORE' 
      and b.data_supplier = 'COMSCORE' 
      and b.current_state = 'MATCHED'
      and a.clientid in('21884504') 
      and iri_week < 1934
order by household_id, date1;

select count(*) 
from daily.daily_unioned a, 
     WH_SUPPLMENTAL.HOUSEHOLD_DEPENDENT_ID_MAP b 
where a.lr_id = b.dependent_id
      and a.clientid in('21884504')
      and iri_week < 1934
      and b.current_state = 'MATCHED'
      and b.retailer = 'COMSCORE' 
      and b.data_supplier = 'EXPERIAN'
;

select count(*) 
from WH_SUPPLMENTAL.HOUSEHOLD_DEPENDENT_ID_MAP b 
where b.retailer = 'COMSCORE' 
      and b.data_supplier = 'COMSCORE'
;

select distinct(retailer) from WH_SUPPLMENTAL.HOUSEHOLD_DEPENDENT_ID_MAP;

select distinct(data_supplier) from WH_SUPPLMENTAL.HOUSEHOLD_DEPENDENT_ID_MAP;
"""


# ========================================================== DATASET AGREGATION =============================================

export JULIA_NUM_THREADS=4


Threads.nthreads()
#addprocs([("10.106.128.212", 1)])
addprocs([("10.106.128.213", 1)])
addprocs([("10.106.128.214", 1)])
addprocs([("10.106.128.215", 1)])
addprocs([("10.106.128.216", 1)])
addprocs([("10.106.128.217", 1)])
addprocs([("10.106.128.218", 1)])
addprocs([("10.106.128.219", 1)])
addprocs([("10.106.128.220", 1)])
addprocs([("10.106.128.221", 1)])
addprocs(1)


#Base.LOAD_CACHE_PATH[1]="/mapr/mapr04p/analytics0001/analytic_users/jpkg/v0.5"
#ENV["JULIA_PKGDIR"]  = "/mapr/mapr04p/analytics0001/analytic_users/jpkg"
#push!(LOAD_PATH, "/mapr/mapr04p/analytics0001/analytic_users/jpkg/v0.5")
#pop!(LOAD_PATH)

#using DataStructures, DataFrames, StatsFuns, GLM , JuMP,NLopt, HDF5, JLD, Distributions, MixedModels, RCall, StatsBase, xCommon, Feather
using DataFrames



#dfd = readtable("/mnt/resource/analytics/trees/exposure.csv",header=false);
dfd = readtable("/mapr/mapr04p/analytics0001/analytic_users/Models/RF/RF.csv",header=true);
#names!(dfdx, [:household_id, :iri_week, :date1, :creative_id, :placement_id, :gross_imps, :val_imps ,:placement, :creative, :publisher] )
#names!(dfd, [:hh_id, :week, :date, :creative_id, :placement_id, :gross_imps, :imps ,:placement, :creative, :publisher] )   
names!(dfd,[Symbol(replace(string(n),"jennie_rf_pos_exposure2_","")) for n in names(dfd)])
rename!(dfd,Symbol("3rdparty"), :thirdrdparty )
rename!(dfd,:household_id, :panid )



#dfd[:household_id]
dformat = Dates.DateFormat("y-m-d"); 
dfd[:date] = map(x-> x=="NULL" ? NA : Date(replace(x, " 00:00:00",""),dformat), dfd[:date1])
dfd = dfd[!isna(dfd[:date]),:]
dfd[:date] = convert(Array{Date},dfd[:date])
dfd[:datelag] = dfd[:date] .- Dates.Day(7)
dfd[:datenum] = map(x->Int64(x),dfd[:date])
dfd[:datenumlag] = map(x->Int64(x),dfd[:datelag])





# ---------------------------------------------------------------------

#Date("2016-07-06",Dates.DateFormat("y-m-d"))

# ----------------- CHUNKS  ------------------
#Date("2016-06-30",Dates.DateFormat("y-m-d"))-Dates.Day(7)
#Date("2016-06-30",Dates.DateFormat("y-m-d"))
#dfd[:date] = map(x-> Date(replace(x, " 00:00:00",""),dformat), dfd[:date1])
#Date(dfd[:date1],dformat)
#Date(dfd[:date1],dformat)
#Date("2016-06-30",dformat)
#dfd=sort(dfd,cols=[:date])
#dfd[:chunks]=-1


panlst = unique(dfd[:panid])
panchunks=DataFrame( panid=panlst,chunks=-1 )
panlen=length(panlst)
chunksize = Int(floor(length(hhlst)/100))
chunknum=1
while (chunknum*chunksize) <= panlen
    s=chunknum*chunksize
    e=s+chunksize
    println("start: ",s,"   end:",e,"   :=  chunk : ",chunknum)
    if e > panlen
        panchunks[s:end,:chunks] = chunknum
    else
        panchunks[s:e,:chunks] = chunknum
    end
    chunknum=chunknum+1
end
panchunks[panchunks[:chunks].==-1,:chunks]=chunknum

countmap(panchunks[:chunks])

dfd2 = join(dfd, panchunks, on = :panid)

root="/mapr/mapr04p/analytics0001/analytic_users/Models/RF/chunks"
writetable(root*"/total.csv", dfd2)
for p in sort(unique(dfd2[:chunks]))
    println(p)
    writetable(root*"/"*string(p)*".csv",  dfd2[dfd2[:chunks].==p,:]    ) 
end



# ---- END CHUNKS ------

#---  Generate Lags ---


export JULIA_NUM_THREADS=4
Threads.nthreads()

#addprocs([("10.106.128.212", 1)])
addprocs([("10.106.128.213", 1)])
addprocs([("10.106.128.214", 1)])
addprocs([("10.106.128.215", 1)])
addprocs([("10.106.128.216", 1)])
addprocs([("10.106.128.217", 1)])
addprocs([("10.106.128.218", 1)])
addprocs([("10.106.128.219", 1)])
addprocs([("10.106.128.220", 1)])
addprocs([("10.106.128.221", 1)])
addprocs(1)

@everywhere insert!(Base.LOAD_CACHE_PATH, 1, "/mapr/mapr04p/analytics0001/analytic_users/jpkg/lib/v0.5")
@everywhere pop!(Base.LOAD_CACHE_PATH)
@everywhere println((Base.LOAD_CACHE_PATH))

using DataFrames
@everywhere using DataFrames
p=1

@everywhere function genLags(p::Int64)  
    brks=[:html5
         ,:static
         ,:native_creative
         ,:video
         ,:content
         ,:contextual
         ,:direct
         ,:prospecting
          ,:retargeting
          ,:behavorial
          ,:native
          ,:predictive
          ,:thirdrdparty
          ,:pmp
          ,:buzzfeed
          ,:hulu
          ,:popsugarus
          ,:rachaelraymag
          ,:shape
          ,:blank
          ,:eatingwell
          ,:meredithcorporation
          ,:realsimple
          ,:turndsp
          ,:cookinglight
          ,:dataxu
          ,:allrecipes
          ,:amazon
          ,:myfitnesspal
          ,:womenshealth
          ,:yummly
         ,:youtube 
        ]
    
    root="/mapr/mapr04p/analytics0001/analytic_users/Models/RF/chunks"  
    dfx = readtable(root*"/"*string(p)*".csv",header=true)
    
    for brk in brks
        lag=Symbol(string(brk)*"_lag")
        println("Generating : ",brk,"  ~ ",lag)
        dfx[lag]=0
        for row in eachrow(dfx)
            #println(r[:video])
            row[lag] = sum(dfx[(dfx[:panid].==row[:panid])&(dfx[:datenum].<=row[:datenum] ) & ( dfx[:datenum].>= row[:datenumlag]  ) ,brk])
        end
    end
    writetable(root*"/"*string(p)*"_out.csv", dfx)
    println("COMPLETE for : ",p,"  in ","/"*string(p)*"_out.csv")
end

"""
@async remotecall_fetch(genLags, 2, 2)
@spawnat 2 genLags(2)

@async remotecall_fetch(genLags, 2, 2)
@async remotecall_fetch(genLags, 3, 3)
@async remotecall_fetch(genLags, 4, 4)
@async remotecall_fetch(genLags, 5, 5)
@async remotecall_fetch(genLags, 6, 6)
@async remotecall_fetch(genLags, 7, 7)
@async remotecall_fetch(genLags, 8, 8)
@async remotecall_fetch(genLags, 9, 9)
@async remotecall_fetch(genLags, 10, 10)
@async remotecall_fetch(genLags, 11, 10)
"""
"""
@async remotecall_fetch(genLags, 2, 11)
@async remotecall_fetch(genLags, 3, 12)
@async remotecall_fetch(genLags, 4, 13)
@async remotecall_fetch(genLags, 5, 14)
@async remotecall_fetch(genLags, 6, 15)
@async remotecall_fetch(genLags, 7, 16)
@async remotecall_fetch(genLags, 8, 17)
@async remotecall_fetch(genLags, 9, 18)
@async remotecall_fetch(genLags, 10, 19)
@async remotecall_fetch(genLags, 11, 20)
"""

x=99
for p in 2:11
    x=x+1
    @async remotecall_fetch(genLags, p,  x)
end


# ---------------- de-chunk files ----------------------
using DataFrames
root="/mapr/mapr04p/analytics0001/analytic_users/Models/RF/chunks"
dfx = readtable(root*"/1_out.csv",header=true); 

for i in 31:60   #2:101
    println("Loading : ",i,"_out.csv")
    t = readtable(root*"/"*string(i)*"_out.csv",header=true); 
    dfx=vcat(dfx,t)
end

#writetable(root*"/1_60_tmp.csv", dfx)


using DataFrames
root="/mapr/mapr04p/analytics0001/analytic_users/Models/RF/chunks"
function consolidateChunks(s::Int64,e::Int64)
    println("Loading : ",string(s),"_out.csv")
    dfx = readtable(root*"/"*string(s)*"_out.csv",header=true); 
    for i in s+1:e   #2:101
        println("Loading : ",i,"_out.csv")
        t = readtable(root*"/"*string(i)*"_out.csv",header=true); 
        dfx=vcat(dfx,t)
    end
    return dfx
end

dfx1 = consolidateChunks(1,30);
dfx2 = consolidateChunks(31,60);
dfx3 = consolidateChunks(61,90);
dfx4 = consolidateChunks(91,101);

dfout=vcat(dfx1,dfx2,dfx3,dfx4)
writetable(root*"/out.csv", dfout)









































# Symbol in expression
a=:xyz
QuoteNode(a)
:(5 - $t)
:( :a in $( :(:a + :b) ) )

s=:ssss
:( :a in $( :(:($s) + $s) ) )


a=:(:aaaa)
b=:(:bbb)
:( :a in $( :($a + $b) ) )


function t(s::Symbol)
           println(s," ~~ ",typeof(s))
       end
t(:xyz)
:(t(:xyz))
ex = :(t(:xyz))


:(t(:xyz))

v=:xyz
:(t( :($v) ))


ai=:aaa
bi=:bbb
a= QuoteNode(a)
b= QuoteNode(b)
:( :a in $( :($a + $b) ) )

macro tst(v, exp)
    println("v : ",v," ~~ ",typeof(v))
    println("exp : ",exp," ~~ ",typeof(exp))
    return exp
end

vi=:v1
@tst vi :(println("hello"))

#---------

#----
@everywhere function genx(p::Int64)
        brks=[:html5
         ,:static
         ,:native_creative
         ,:video
         ,:content
         ,:contextual
         ,:direct
         ,:prospecting
          ,:retargeting
          ,:behavorial
          ,:native
          ,:predictive
          ,:thirdrdparty
          ,:pmp
          ,:buzzfeed
          ,:hulu
          ,:popsugarus
          ,:rachaelraymag
          ,:shape
          ,:blank
          ,:eatingwell
          ,:meredithcorporation
          ,:realsimple
          ,:turndsp
          ,:cookinglight
          ,:dataxu
          ,:allrecipes
          ,:amazon
          ,:myfitnesspal
          ,:womenshealth
          ,:yummly
         ,:youtube 
        ]
    dfx = readtable("/mapr/mapr04p/analytics0001/analytic_users/Models/RF/chunks/"*string(p)*".csv",header=true)
    println("xyz : ",p)
end
@spawnat 5  genx(64)


p=1
root="/mapr/mapr04p/analytics0001/analytic_users/Models/RF/chunks"   
dfx = readtable(root*"/"*string(p)*".csv",header=true)
y=:net_price
#----



  brk=:youtube
  lag=:youtube_lag
  dfx[lag]=0
  for row in eachrow(dfx)
        #println(r[:video])
        row[lag] = sum(dfx[(dfx[:panid].==row[:panid])&(dfx[:datenum].<=row[:datenum] ) & ( dfx[:datenum].>= row[:datenumlag]  ) ,brk])
  end

dfx[(dfx[brk].!=dfx[lag]) ,[:panid,:date,:datelag,:video_lag,:datenum,:datenumlag,brk]]
t = dfx[5634,[:panid,:date,:datelag,lag,:datenum,:datenumlag,brk]]      #...... 1321364407 │ "2016-07-18" │ "2016-07-11" │ 2         │ 736163  │ 736156
t = dfx[(dfx[:datenum].==736174)&(dfx[:panid].==1338983878),[:panid,:date,:datelag,lag,:datenum,:datenumlag,brk]]
dfx[(dfx[:panid].==t[:panid][1]) ,brk]
dfx[(dfx[:panid].==t[:panid][1])&(dfx[:datenum].<=t[:datenum][1] ) & (dfx[:datenum].>= t[:datenumlag][1]  ) ,brk]
sum(dfx[(dfx[:panid].==t[:panid][1])&(dfx[:datenum].<=t[:datenum][1]) & (  dfx[:datenum].>= t[:datenumlag][1]  ) ,brk])

# ----------------


dfd[1,:panid]
dfd[1,[:panid,:date,:datelag]]       == 1305262312 │ 2016-07-06 │ 2016-06-29 
dfd[dfd[:panid].==1305262312,:]
sum(dfd[dfd[:panid].==1305262312,:video])

dfd[(dfd[:panid].==1305262312)&(dfd[:date].<=Date("2016-07-06",Dates.DateFormat("y-m-d")) ) & (  dfd[:date].>=Date("2016-06-29",Dates.DateFormat("y-m-d"))  ) ,:]


dfd[1,[:panid,:date,:datelag,:datenum,:datenumlag]]
sum(dfd[(dfd[:panid].==1305262312)&(dfd[:datenum].<=736151 ) & (  dfd[:datenum].>= 736144  ) ,:video])


dfd[:video_lag]=0
for row in eachrow(dfd)
    #println(r[:video])
    row[:video_lag] = sum(dfd[(dfd[:panid].==row[:panid])&(dfd[:datenum].<=row[:datenum] ) & (  dfd[:datenum].>= row[:datenumlag]  ) ,:video])
end


# ---------------





#hhlen=length(hhlst)
hhlen = Int(floor(length(hhlst)/10000))
for hhcnt in 0:hhlen
    lb=(hhcnt*50000)+1
    if hhcnt < hhlen
        ub=(hhcnt+1)*50000
        println("chuncking : (",lb,":",ub,")")
        #dfd[findin(dfd[:panid],panids) ,:iso] = true
        dfd[ findin(dfd[:panid],hhlst[lb:ub])  ,:chunks] = hhcnt+1
    else
        println("chuncking : (",lb,":end)")
        #dfd[ findin(dfd[:panid],hhlst[lb:end])  ,:chunks] = hhcnt+1
        dfd[dfd[:chunks].<0,:chunks]=hhcnt+1
    end
end



#dfd[:dateX] = map(x->    Date(string(x),Dates.DateFormat("yyyymmdd"))   ,dfd[:date])
#dfd[:dateX] = convert(Array{Date}, dfd[:dateX])

#lag_cnt=Dict(:one=>1,:four=>4,:eight=>8,:sixteen=>16)

for brk in [:placement, :creative, :publisher]
    for lvl in unique(dfd[brk])
        for lag in [1,4] #[:one,:four,:eight]
            col=Symbol(string(brk)*"_"*replace(string(lvl)," ","_")*"_"*string(lag))
            println(col)
            dfd[col]=0
        end
    end
end

for chunk in 1:2   #maximum(dfd[:chunks])
    for row in eachrow(dfd[dfd[:chunks].==chunk,:])
        println(row[:hh_id]," ~~~ ",row[:date])
    end
end



x=by(dfd,[:date,:hh_id], df-> sum(     df[ ()&()  :imps]       ))
x=by(dfd,[:date,:hh_id], df-> sum(  df[ ()&()  :imps]       ))


y=by(dfd, [:date,:hh_id,:publisher], nrow)

    
    
    
by(dfd, [:date,:hh_id]) do df
    DataFrame(m = mean(df[:PetalLength]), s² = var(df[:PetalLength]))
end


gb = groupby(dfd, [:date])


by(iris, :Species) do df
    DataFrame(m = mean(df[:PetalLength]), s² = var(df[:PetalLength]))
end


#=========================== END =============================

x=by(dfd,:hh_id, df-> sum(df[:imps] ))
or : x=by(dfd,:hh_id, df-> sum(df[:gross_imps] ))
x[x[:x1].>0,:]
x[x[:hh_id].==1000120003,:]


function oliers(dfd::DataFrame, k::Symbol, c::Symbol)
    m = median(dfd[c])
    #dfd[c_med1] = abs(dfd[c]-m)
    #MAD=median(dfd[c_med1])
    MAD = median(abs(dfd[c]-m))
    dfd[c_med2] =  ((dfd[c]-m)*0.6745) / MAD 

    dfd_zsc = dfd[abs(dfd[c_med2]) .< 3.5 ,:]
    df_in_pout =  join(df_in, dfd_zsc, on = [ k, k ], kind = :inner)
end
oliers(dfd,:household_id, :val_imps)


# by(dfd, :, df -> sum(df[:PetalLength]))
function Pre_out(df_in::DataFram)
    df_cat_pre = df_in[df_in[:Buyer_Pre_P0] .==1 , [:Prd_0_Net_Pr_PRE,:experian_id]]
    median_df = median(df_cat_pre[:Prd_0_Net_Pr_PRE])
    df_cat_pre[:Prd_0_Net_Pr_PRE_med1] = abs(df_cat_pre[:Prd_0_Net_Pr_PRE]-median_df)
    MAD=median(df_cat_pre[:Prd_0_Net_Pr_PRE_med1])
    df_cat_pre[:Prd_0_Net_Pr_PRE_med2] = (0.6745*(df_cat_pre[:Prd_0_Net_Pr_PRE]-median_df))/MAD
    df_cat_pre_zsc = df_cat_pre[abs(df_cat_pre[:Prd_0_Net_Pr_PRE_med2]) .< 3.5 ,:]
    df_in_pout =  join(df_in, df_cat_pre_zsc, on = [ :experian_id, :experian_id ], kind = :inner)
end
Pre_out(df_in)


# --- parallel RF

function build_forest(labels, features, nsubfeatures, ntrees, ncpu=1)
                  if ncpu > nprocs()
                       addprocs(ncpu - nprocs())
                  end
                  Nlabels = length(labels)
                  Nsamples = int(0.7 * Nlabels)
                  forest = @parallel (vcat) for i in [1:ntrees]
                      inds = rand(1:Nlabels, Nsamples)
                      build_tree(labels[inds], features[inds,:], nsubfeatures)
                  end
                  return [forest]
              end

+++++++++Distribute Macro ++++++++++++++++++++++++++++
function sync_add(r)
    spawns = get(task_local_storage(), :SPAWNS, ())
    if spawns !== ()
        push!(spawns[1], r)
        if isa(r, Task)
            tls_r = get_task_tls(r)
            tls_r[:SUPPRESS_EXCEPTION_PRINTING] = true
        end
    end
    r
end

spawnat(p, thunk) = sync_add(remotecall(thunk, p))

macro spawnat(p, expr)
    expr = localize_vars(esc(:(()->($expr))), false)
    :(spawnat($(esc(p)), $expr))
end

macro run(p, expr)
    expr = localize_vars(esc(:(()->($expr))), false)
    :(spawnat($(esc(p)), $expr))
end
+++++++++++++++++++++++++++++++++++++


================== XGBoost ============================================= https://www.kaggle.com/wacaxx/rossmann-store-sales/julia-xgboost-starter-code
using BinDeps, DataFrames, XGBoost
#using Dates

y=convert(Array{Float32},dfd[:dol_per_trip_pre_p1] )
x=dfd[setdiff(names(dfd),[:dol_per_trip_pre_p1])]
#Define target
#y = convert(Array{Float32}, train[:Sales])

#Transform data to XGBoost matrices
#trainArray = convert(Array{Float32},  x[:, vcat(numericalColumns, categoricalColumns)])
#testArray = convert(Array{Float32}, test[:, vcat(numericalColumns, categoricalColumns)])
#dtrain = DMatrix(trainArray, label = costTrainingLog)
#dtest = DMatrix(testArray)

num_round = 250
param = ["eta" => 0.2, "max_depth" => 20, "objective" => "reg:linear", "silent" => 1]

XGBoostModel = xgboost(dtrain, num_round, param = param)


#Predictions using test data
preds = predict(XGBoostModel, dtest)
#Round to zero closed stores
preds[closedStoreIdx] = 0

#Write Results
sampleSubmission = DataFrame(Id = tesdIdx, Sales = preds)

==================  TREES WORKING ==================== examples https://github.com/bensadeghi/DecisionTree.jl
using DecisionTree
y=dfd[:dol_per_trip_pre_p1]
x=dfd[setdiff(names(dfd),[:dol_per_trip_pre_p1])]

model = build_tree(Array(y), Array(x), 5)
or 
m2 = build_forest(Array(y), Array(x), 2, 10, 5, 0.7)

apply_tree(model, Array(x))


xtest=x[1:1000,:]
xtrain=x[1000:end,:]
ytest=y[1:1000]
ytrain=y[1000:end]
model = build_tree(Array(ytrain), Array(xtrain), 5)
res = apply_tree(model, Array(xtest))
map((x,y)->x-y, res,ytest)

r2 = nfoldCV_tree(Array(ytrain), Array(xtrain), 3, 5)

m2 = build_forest(Array(ytrain), Array(xtrain), 2, 10, 5, 0.7)
res = apply_tree(m2, Array(xtest))

====================== END ===========================

df_in[:Dol_per_Trip_PRE_P1]

setdiff(names(df_in),:Dol_per_Trip_PRE_P1)


y=df_in[:Dol_per_Trip_PRE_P1]
x=df_in[setdiff(names(df_in),[:Dol_per_Trip_PRE_P1])]
#model = build_forest(y,x, 20, 50, 1.0)
model = build_tree(y, Array(x),20, 50, 1.0);

using DecisionTree

model = build_forest(df_in[:Dol_per_Trip_PRE_P1], df_in[[setdiff(names(df_in),:Dol_per_Trip_PRE_P1)]], 20, 50, 1.0)

x=Array(x[setdiff(names(x),[:core_based_statistical_areas,:person_1_birth_year_and_month])])

model = build_tree(Array(y), Array(x),20, 50, 1.0);






========================================================
#https://github.com/dmlc/XGBoost.jl/blob/master/demo/basic_walkthrough.jl

using XGBoost

function readlibsvm(fname::ASCIIString, shape)
    dmx = zeros(Float32, shape)
    label = Float32[]
    fi = open(fname, "r")
    cnt = 1
    for line in eachline(fi)
        line = split(line, " ")
        push!(label, float(line[1]))
        line = line[2:end]
        for itm in line
            itm = split(itm, ":")
            dmx[cnt, int(itm[1]) + 1] = float(int(itm[2]))
        end
        cnt += 1
    end
    close(fi)
    return (dmx, label)
end

train_X, train_Y = readlibsvm("/home/rmadmin/.julia/v0.4/XGBoost/data/agaricus.txt.train", (6513, 126))
test_X, test_Y = readlibsvm("/home/rmadmin/.julia/v0.4/XGBoost/data/agaricus.txt.test", (1611, 126))

=============================
using DecisionTree

#Train random forest with
#20 for number of features chosen at each random split,
#50 for number of trees,
#and 1.0 for ratio of subsampling.
model = build_forest(yTrain, xTrain, 20, 50, 1.0)


using DataStructures, DataFrames, StatsFuns, GLM , JuMP,NLopt, HDF5, JLD, Distributions, MixedModels, RCall, StatsBase, xCommon

function loadDF()
    cd("/media/u01/analytics/scoring/CDW5_792/")
    df_data = readtable("csv_final_cdw5_792.csv",header=false);
    df_h = readtable("Headers_cdw5.csv",header=false);
    names!(df_data, convert(Array{Symbol}, df_h[:x1]) )
end
df_in=loadDF()






# --===========================================================================================================================================
# --===================================== SQL =================================================================================================
# --===========================================================================================================================================





select count(a.household_id), count(distinct a.household_id), b.operating_company
from wh_supplmental.household_dependent_id_map a 
join wh_fsp.partitioned_fsp_wkly_fact b 
     on a.dependent_id = b.card_num 
     and UPPER(trim(b.Operating_company)) = trim(UPPER(case when a.retailer='FOODLION' then 'FOOD LION' else a.retailer end ))
where a.current_state = 'MATCHED' 
  and a.household_type = 'EXPERIAN' 
  and a.dependent_type = 'FSP_CARD' 
  and TM_DIM_KEY_WEEK>=1909 
group by b.operating_company
"""
149243  17012   FREDS
196025906       7033205 RITEAID
330798388       994861  WEGMANS
38008898        221081  KEYFOOD
6020735400      15722602        KROGER
16025680        100830  SUPERVALU
302291297       2137143 BJS
"""



hive -e 'SELECT distinct household_id FROM WH_SUPPLMENTAL.HOUSEHOLD_DEPENDENT_ID_MAP where household_type = "EXPERIAN" and dependent_type = "FSP_CARD" and  dependent_id is not null' | sed 's/[\t]/,/g' > /mapr/mapr04p/analytics0001/analytic_users/mddak/fsp_ids_retailer2.csv


set hive.cli.print.header=true; 
SELECT retailer, count(distinct household_id) as ids FROM WH_SUPPLMENTAL.HOUSEHOLD_DEPENDENT_ID_MAP where household_type = "EXPERIAN" and dependent_type = "FSP_CARD" and  dependent_id is not null GROUP BY retailer;

hadoop fs -put /mapr/mapr04p/analytics0001/analytic_users/mddak/Experian/IRI_Key_Food_Shipment.txt /externaldata01/prd/experian/xwalk/raw/


set hive.cli.print.header=true; select * from wh_fsp.partitioned_fsp_wkly_fact limit 10;
hive -e 'set hive.cli.print.header=true; select * from MEDIA_LIFT_CAMPAIGN.lift_product WHERE registration_request_id=931 limit 10' | sed 's/[\t]/,/g' > /mapr/mapr04p/analytics0001/analytic_users/mddak/lift_product_example10.csv
set hive.cli.print.header=true; select * from MEDIA_LIFT_CAMPAIGN.lift_product_item_filter_map WHERE registration_request_id=931 limit 10;
set hive.cli.print.header=true; select * from MEDIA_LIFT_CAMPAIGN.POS_WKLY_FACT POS WHERE registration_request_id=931 limit 10; 


set hive.cli.print.header=true; select * from Jennieo6.Kantar_FSP_POS_Jennieo6_931 limit 10;


