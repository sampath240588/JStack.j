using DataStructures, DataArrays , DataFrames, StatsFuns, GLM , Distributions, MixedModels, StatsBase, StatLib, JSON, NamedArrays
 
const cfgDefaults=OrderedDict( :P2_Competitor => true
                        ,:pvalue_lvl => 0.20  #pvalue_lvl = 0.20 
                        ,:excludedBreaks => String[]    #["estimated_hh_income","hh_age","number_of_children_in_living_un","person_1_gender"]
                        ,:excludedLevels => ["none"]
                        ,:excludedKeys => String[]
                        ,:exposed_flag_var => :exposed_flag
                        ,:sigLevel => "0.2"
                        ,:random_demos => [:estimated_hh_income,:hh_age,:number_of_children_in_living_Un,:person_1_gender]
                        ,:random_campaigns => []
                        ,:dropvars => [:exposed_flag]
                        ,:scoring_vars => [:Prd_1_Net_Pr_PRE,:Prd_1_Net_Pr_POS,:Buyer_Pos_P0,:Buyer_Pre_P0]
                        ,:occ_y_var => :Trps_POS_P1
                        ,:occ_logvar => :Trps_PRE_P1
                        ,:dolocc_y_var => :Dol_per_Trip_POS_P1
                        ,:dolocc_logvar => :Dol_per_Trip_PRE_P1
                        ,:pen_y_var => :Buyer_Pos_P1
                        ,:pen_logvar => :Buyer_Pre_P1
                        ,:TotalModelsOnly=>false
                       )

root = pwd()


#root="/mnt/resource/analytics/models"
#Default
cfgDefaults[:random_demos] = [:estimated_hh_income,:hh_age,:number_of_children_in_living_Un,:person_1_gender]
cfgDefaults[:random_campaigns] = [:Publisher_Fct1,:Targeting_Fct1]
cfgDefaults[:exposed_flag_var] = :exposed_flag_new


# --------------------------------- CAMPAIGNS ----------------------------------------
#NatC : 
cfgDefaults[:random_demos] = [:estimated_hh_income,:hh_age,:person_1_gender]
cfgDefaults[:random_campaigns] = [:Targeting_fct,:Media_Source_fct,:Media_Type_fct,:Publisher_fct,:Creative_fct]      
cfgDefaults[:exposed_flag_var] = :exposed_flag_new
root="/mnt/resource/analytics/models/NatChoice"

#-- CDW
cfgDefaults[:random_demos] = [:estimated_hh_income,:hh_age,:number_of_children_in_living_Un,:person_1_gender]
cfgDefaults[:random_campaigns] = [:Publisher_Fct1,:Targeting_Fct1]
cfgDefaults[:exposed_flag_var] = :exposed_flag_new
root="/mnt/resource/analytics/models/CDW"


# JennieO - :creative :publisher1 :placement
cfgDefaults[:random_demos] = [:estimated_hh_income,:hh_age,:number_of_children_in_living_Un,:person_1_gender]
cfgDefaults[:random_campaigns] = [:creative,:publisher1,:placement]
cfgDefaults[:exposed_flag_var] = :exposed_flag_new
root="/mnt/resource/analytics/models/Jenny-o"


# Rev7 
cfgDefaults[:random_demos] = [:hh_age,:estimated_hh_income,:number_of_children_in_living_Un,:person_1_gender]
cfgDefaults[:random_campaigns] = [:creativename, :media_source,:media_type,:publisher,:targeting]
cfgDefaults[:exposed_flag_var] = :exposed_flag_new
root="/mnt/resource/analytics/models/rev"


#ALL#10  - /mapr/mapr04p/analytics0001/analytic_users/Models/All_10_944/All_10_Modeling_944/Output 
root="/mnt/resource/analytics/models/ALL#10"
cfgDefaults[:random_demos] = [:estimated_hh_income,:hh_age,:number_of_children_in_living_Un,:person_1_gender]
cfgDefaults[:random_campaigns] = [:creative,:ad_type,:publisher,:frequency_type]
cfgDefaults[:exposed_flag_var] = :exposed_flag_new


#ALL#11  - /mapr/mapr04p/analytics0001/analytic_users/Models/All_11_983/All_11_983_Modeling/Output
root="/mnt/resource/analytics/models/ALL#11"
cfgDefaults[:random_demos] = [:estimated_hh_income, :hh_age, :number_of_children_in_living_Un, :person_1_gender]
cfgDefaults[:random_campaigns] = [:creative,:ad_type,:publisher,:frequency_type]
cfgDefaults[:exposed_flag_var] = :exposed_flag_new


#CDW#6  - /mapr/mapr04p/analytics0001/analytic_users/Models/CDW_6_949/CDW6_modeling/Output7”
root="/mnt/resource/analytics/models/CDW#6"
cfgDefaults[:exposed_flag_var] = :exposed_flag_new
cfgDefaults[:random_demos] = [:estimated_hh_income, :hh_age, :number_of_children_in_living_Un, :person_1_gender]
cfgDefaults[:random_campaigns] = [:viewability_break, :Publisher_Fct1,:Targeting_Fct1]


#HormelChili#8  - /mapr/mapr04p/analytics0001/analytic_users/Models/Hormel_Chili_8/Hormel_Chili8_model/output
root="/mnt/resource/analytics/models/HormelChili#8"
cfgDefaults[:exposed_flag_var] = :exposed_flag_new
cfgDefaults[:random_demos] = [:estimated_hh_income, :hh_age, :number_of_children_in_living_Un, :person_1_gender]
cfgDefaults[:random_campaigns] = [:publisher,:media_type,:creativename]


#Rev#8  -        "/mapr/mapr04p/analytics0001/analytic_users/Models/Hormel_Rev_8_967/Hormel_Rev_8_Data/Output1/"
root="/mnt/resource/analytics/models/Rev#8"
cfgDefaults[:exposed_flag_var] = :exposed_flag_new
cfgDefaults[:random_demos] = [:estimated_hh_income, :hh_age, :number_of_children_in_living_Un, :person_1_gender]
cfgDefaults[:random_campaigns] = [:Creative_fct_1,:Targeting_fct_1,:Publisher_fct_1,:MediaType_fct_1,:MediaSource_fct_1]



# ------------------------------- END CAMPAIGNS --------------------------------------



cd(root)
function loadDF()
    #cd("/media/u01/analytics/scoring/Healthy/modeling5/")
    #df_data = readtable("csv_final_healthychoice1_I3.csv",header=true); #lowercase(df_data)
    #df_h = readtable("Headers_healthy_choice3.csv",header=false); #lowercase(df_data)
    #names!(df_data, convert(Array{Symbol}, df_h[:x1]) ) 
  
    df_data = readtable(root*"/orig.csv",header=false);
    df_h = readtable(root*"/origHead.csv",header=false);  
    names!(df_data, convert(Array{Symbol}, df_h[:x1]) )
end
df_in=loadDF()



function Pre_out(df_in::DataFrame)
     df_cat_pre = df_in[df_in[:Buyer_Pre_P0] .==1 , [:Prd_0_Net_Pr_PRE,:experian_id]]
     df_cat_pos = df_in[(df_in[:Buyer_Pre_P0] .==0) & (df_in[:Buyer_Pos_P0] .==1) , [:experian_id]]
     median_df = median(df_cat_pre[:Prd_0_Net_Pr_PRE])
     df_cat_pre[:Prd_0_Net_Pr_PRE_med1] = abs(df_cat_pre[:Prd_0_Net_Pr_PRE]-median_df)
     MAD=median(df_cat_pre[:Prd_0_Net_Pr_PRE_med1])
     df_cat_pre[:Prd_0_Net_Pr_PRE_med2] = (0.6745*(df_cat_pre[:Prd_0_Net_Pr_PRE]-median_df))/MAD
     df_cat_pre_zsc = df_cat_pre[abs(df_cat_pre[:Prd_0_Net_Pr_PRE_med2]) .< 3.5,:]
     df_cat_pre_zsc_1 = df_cat_pre_zsc[:,[:experian_id]]
     df_cat_pre_zsc_f = vcat(df_cat_pos,df_cat_pre_zsc_1)
     df_in_pout =  join(df_in, df_cat_pre_zsc_f, on =  :experian_id , kind = :inner);
end
df_in = Pre_out(df_in);



function isValid(df_data::DataFrame,cfg::OrderedDict)
    function checkValid(iarr::Array{Symbol})  length(setdiff(iarr, names(df_data))) > 0 ? false : true end
    !checkValid(cfg[:all_mandatory_vars]) ? error("ERROR: Not all mandatory_vars in dataset ") : println("VALID : mandatory_vars") 
    !checkValid(cfg[:scoring_vars]) ? error("ERROR: Not all scoring_vars in dataset ") : println("VALID : scoring_vars") 
end






function getCFG(df_in::DataFrame)
    cfg=StatLib.loadCFG(cfgDefaults, pwd()*"/app.cfg")
    #cfg[:exposed_flag_var] = :exposed_flag_new                  # Go In app.cfg
    #cfg[:random_campaigns] = [:Publisher_Fct1,:Targeting_Fct1]  # Go In app.cfg

    cfg[:allrandoms] = vcat(cfg[:random_demos],cfg[:random_campaigns])
    #cfg[:ProScore] = grep1("MODEL",names(df_in))   
    ps = filter(x->contains(string(x), "MODEL"), names(df_in)) # Set the ProScore variable in the dataset   #push!(cfg[:all_mandatory_vars], cfg[:ProScore])
    cfg[:ProScore] = length(ps) > 0 ? ps[1] : :MISSING_MODEL_VARIABLE_IN_DATA
    cfg[:num_products] = length( grep("Buyer_Pre_P",names(df_in)))-1  # get number of products in the data
    #rework vars after loading cfg
    cfg[:random_campaigns] = intersect(cfg[:random_campaigns],names(df_in))
 
    cfg[:occ_logvar_colname] = Symbol("LOG_"*string(cfg[:occ_logvar]))
    cfg[:dolocc_logvar_colname] = Symbol("LOG_"*string(cfg[:dolocc_logvar]))
    cfg[:pen_logvar_colname] = Symbol("LOG_"*string(cfg[:pen_logvar]))
    
    cfg[:all_mandatory_vars] = vcat( [:experian_id,
                                      cfg[:pen_y_var],cfg[:pen_logvar],
                                      cfg[:occ_y_var], cfg[:occ_logvar],
                                      cfg[:dolocc_y_var], cfg[:dolocc_logvar]
                                     ]
                                     , cfg[:random_demos]
                                   )
    return cfg
end
cfg=getCFG(df_in)

isValid(df_in,cfg)



function reworkCFG!(df_in::DataFrame,cfg::OrderedDict)
    xflags=[cfg[:exposed_flag_var],:exposed_flag,:Nonbuyer_Pre_P1]  # :Nonbuyer_Pre_P1 no longer used
    features = setdiff(names(df_in),xflags)
    if :state in names(df_in)
        cfg[:xVarsDemos]=vcat([:experian_id, :banner, :Prd_0_Qty_PRE],features[findfirst(features, :state):findfirst(features, :Mosaic)])   # exclude demos between....
    else
        cfg[:xVarsDemos]=Symbol[]
    end
    cfg[:xVarsDemos]=setdiff(cfg[:xVarsDemos],[:person_1_gender,:number_of_children_in_living_Un]) # exclude person, # from non-used demos
    cfg[:xVarsPost] = grep(["POS","Pos","Buyer_Pre_","Nonbuyer_Pre_"],features)   #exclude POST variables
    cfg[:iVarsPREPOS] = grep(["PRE_POS","Pre_Pos"],features) 
    features=setdiff(features, setdiff(  vcat(cfg[:xVarsDemos],cfg[:xVarsPost]) ,cfg[:iVarsPREPOS])  )
    cfg[:xVarsP0] =  setdiff(grep("0", features ),[cfg[:ProScore]] ) #exclude category variables(exclude P0's)  
    features=setdiff(features,cfg[:xVarsP0])
    cfg[:xVarsReports] = grep(["Perc_","Pr_per_"],features)  #exclude Reporting vars
    features=setdiff(features,cfg[:xVarsReports])
    cfg[:xvars] = vcat(xflags,cfg[:xVarsDemos], cfg[:xVarsPost],cfg[:xVarsP0], cfg[:xVarsReports],cfg[:dropvars])
    features=setdiff(features,cfg[:dropvars])
    cfg[:ivars] = vcat(cfg[:iVarsPREPOS],cfg[:all_mandatory_vars],cfg[:scoring_vars])
    cfg[:ALL_vars_to_exclude] = setdiff(cfg[:xvars], cfg[:ivars])
    df_in[:group] = df_in[cfg[:exposed_flag_var]]
    df_in[:panid] = df_in[:experian_id] 
    features = setdiff(unique(vcat(features,cfg[:iVarsPREPOS],cfg[:all_mandatory_vars],cfg[:scoring_vars],[:group,:panid])  ) ,[:experian_id] )
    cfg[:negativevars] = grep(vec(hcat([["P"*string(i),string(i)*"_"] for i=3:cfg[:num_products]]...)),features)   # get variables that need to have negative sign
    cfg[:positivevars] = grep(["P1", "1_","MODEL"], features)    # get variables that need to have positive sign
    if cfg[:P2_Competitor] == true  
        cfg[:negativevars] = unique(vcat(cfg[:negativevars],grep(["P2","2_"],features))) 
    else
        cfg[:positivevars] = unique(vcat(cfg[:positivevars],grep(["P2","2_"],features))) 
    end
    return df_in[features]
end

dfd = reworkCFG!(df_in,cfg)



#######################################
#--------- Data Manipulation ---------#
#######################################

function data_Prep(dfd::DataFrame, cfg::OrderedDict)
    dfd[:isO]=false
    dfd[dfd[:number_of_children_in_living_Un].>=4,:number_of_children_in_living_Un] = 4  # aggregate #of children for 4+ L_114
    if typeof(dfd[:group]) in [DataArray{String,1}] 
        dfd[ findin(dfd[:group],["//N","\\N"]), :group] = "0" 
        dfd[DataArrays.isna(dfd[:group]), :group]="0"
        dfd[:group] = [parse(Int64,s) for s = dfd[:group]]
    else
        dfd[ DataArrays.isnan(dfd[:group]), :group] = 0
    end
    vars = DataFrame(names=names(dfd),eltypes=eltypes(dfd))
    #for c in setdiff(y,cfg[:allrandoms]) # set variables as.numeric and replace NA's with zero
    for c in setdiff(vars[findin(vars[:eltypes],[String]),:names],cfg[:allrandoms])
        print("Convert String: String->Numeric: ",c)
        try dfd[c] = map( x -> DataArrays.isna.(x) ?  NaN : convert(Float64, x)  , dfd[c]) catch e println("  (failed)") end  #NA to NaN
        try dfd[c] = convert(Array{Float64}, dfd[c]) catch e end   # To Float64
    end
    vars = DataFrame(names=names(dfd),eltypes=eltypes(dfd))
    for c in setdiff(vars[findin(vars[:eltypes],[Float64]),:names],cfg[:allrandoms])  # replace NaN's with zero for numeric variables 
        println("Replace Float64 NaN (0.0): ",c)
        dfd[ DataArrays.isnan(dfd[c]), c] = 0.0
    end
    for c in  setdiff(vars[findin(vars[:eltypes],[Int64]),:names],cfg[:allrandoms])    
        println("Replace Int64 NaN (0) : ",c)
        dfd[ DataArrays.isnan(dfd[c]), c] = 0
    end
    # NOTE : Do QCs here
    dfd[dfd[:person_1_gender].=="U",:isO] = true   # remove HHs with no gender info
    #dfd[dfd[:person_1_gender].=="U",:whyO] = "person_1_gender=U; remove HHs with no gender info"
    dfd[findin(dfd[:estimated_hh_income],["U","L"]),:estimated_hh_income]="L" # aggregate U and L levels of hh income

    for r in cfg[:random_campaigns]    # check and drop exposed HHs with no publisher info or non-exposed HHs with publisher info
        dfd[findin(dfd[r],["\\N","NULL","0","NONE"])  ,r] ="none"
        #dfd[ (dfd[:isO].==false) & (dfd[r].!="none") & (dfd[:group].==0) ,:whyO] = "non exposed HHs with publisher info"
        dfd[ (dfd[:isO].==false) & (dfd[r].!="none") & (dfd[:group].==0) ,:isO] = true
        #println(r," non exposed HHs with publisher info : ",nrow(dfd[dfd[:whyO].=="non exposed HHs with publisher info",:] )   )
        #dfd[ (dfd[:isO].==false) & (dfd[r].=="none") & (dfd[:group].!=0) ,:whyO] = "exposed HHs with no publisher info"   
        dfd[ (dfd[:isO].==false) & (dfd[r].=="none") & (dfd[:group].!=0) ,:isO] = true
        #println(r," exposed HHs with no publisher info : ",nrow(dfd[dfd[:whyO].=="exposed HHs with no publisher info",:] )   )
    end
    # segments for outliers detection
    dfd[:data_NB_NE_B] = false
    dfd[ (dfd[:Buyer_Pre_P1].==0 ) & (dfd[:group].==0 ) & (dfd[:Buyer_Pos_P1].==1 ) ,:data_NB_NE_B] = true
    dfd[:data_B_E_NB] = false
    dfd[ (dfd[:Buyer_Pre_P1].==1 ) & (dfd[:group].==0 ) & (dfd[:Buyer_Pos_P1].==0 ) ,:data_B_E_NB] = true
    dfd[:pen_reduction] = false
    dfd[ (dfd[:Buyer_Pre_P1].==1) & (dfd[:group].==0) & (dfd[:Buyer_Pos_P1].==0 )  ,:pen_reduction] = true
    dfd[:occ_reduction] = false
    dfd[ (dfd[:group].==0) & (dfd[:Buyer_Pos_P1].==1) & (dfd[:Trps_POS_P1].< dfd[:Trps_PRE_P1] ) ,:occ_reduction] = true
    dfd[:dolocc_reduction] = false
    dfd[  (dfd[:group].==0) & (dfd[:Buyer_Pos_P1].==1) & (dfd[:Dol_per_Trip_POS_P1].< dfd[:Dol_per_Trip_PRE_P1] )  , :dolocc_reduction] = true
    return dfd[dfd[:isO].==false, : ]   #[setdiff(names(dfd),[:isO,:whyO])] 
end

dfd = data_Prep(dfd, cfg);




function MatchMe(dfd::DataFrame,cfg::OrderedDict)
    df=dfd[dfd[:isO].==false,:]
    df_exp     = df[df[:group].==1,:]
    df_unexp   = df[df[:group].==0,:]
    df_exp_dim   = nrow(df_exp)
    df_unexp_dim = nrow(df_unexp)
    new_unexp_dim = df_unexp_dim*(df_unexp_dim>2000000 ? 0.3 : df_unexp_dim>1000000 ? 0.4 : df_unexp_dim>750000 ? 0.6 : 0.7)
    if length(string(cfg[:ProScore])) == 0
        df_unexp_1 =  df[(df[:group].==0)&(df[:Buyer_Pre_P1].==1),:]
        df_unexp_0 =  df[(df[:group].==0)&(df[:Buyer_Pre_P1].==0),:]
        
        df_exp_1_dim = nrow(df[(df[:group].==1)&(df[:Buyer_Pre_P1].==1),:])
        df_exp_0_dim = nrow(df[(df[:group].==1)&(df[:Buyer_Pre_P1].==0),:])
        df_unexp_1_dim = nrow(df_unexp_1)
        df_unexp_0_dim =  nrow(df_unexp_0)     
        dim_sample0 = round(Int64, (new_unexp_dim-df_exp_dim  ) / (1+(df_exp_1_dim / df_exp_0_dim)) )
        dim_sample1 = round(Int64, new_unexp_dim - df_exp_dim - dim_sample0)    
        
        new_df_unexp_1 = df_unexp_1[sample(1:size(df_unexp_1,1), dim_sample1 ),:]
        new_df_unexp_0 = df_unexp_0[sample(1:size(df_unexp_0,1), dim_sample0 ),:]
        dfd_sample  = vcat(df_exp,new_df_unexp_1,new_df_unexp_0)
    elseif length(string(cfg[:ProScore])) > 0    
        sample_control_data=similar(df_unexp, 0)
        for (key, value) in countmap(df_exp[cfg[:ProScore]])
            sample_dim=round(Int64,new_unexp_dim*(value/df_exp_dim))
            temp_data = df_unexp[df_unexp[cfg[:ProScore]].==key,:]
            if sample_dim > size(temp_data,1) sample_dim = size(temp_data,1) end  #??
            samp_data = temp_data[sample(1:size(temp_data,1), sample_dim, replace=false),:]
            sample_control_data = vcat(sample_control_data,    samp_data   )
        end
        
        sample_data = vcat(sample_control_data,df_exp)
        sample_df_unexp = sample_data[sample_data[:group].==0,:]
        sample_df_exp   = sample_data[sample_data[:group].==1,:]
        sample_df_unexp_1 = sample_data[(sample_data[:group].==0)&(sample_data[:Buyer_Pre_P1].==1),:]  
        sample_df_unexp_0 = sample_data[(sample_data[:group].==0)&(sample_data[:Buyer_Pre_P1].==0),:]    
        sample_df_exp_1 =  sample_data[(sample_data[:group].==1)&(sample_data[:Buyer_Pre_P1].==1) ,:]
        sample_df_exp_0 =  sample_data[(sample_data[:group].==1)&(sample_data[:Buyer_Pre_P1].==0) ,:]
        sample_df_unexp_1_dim = nrow(sample_df_unexp_1)
        sample_df_unexp_0_dim = nrow(sample_df_unexp_0)
        sample_df_exp_1_dim = nrow(sample_df_exp_1)
        sample_df_exp_0_dim = nrow(sample_df_exp_0)
        dim_sampleA = round(Int64,(sample_df_exp_1_dim/sample_df_exp_0_dim)*sample_df_unexp_0_dim)
        dim_sampleB = round(Int64,(sample_df_exp_0_dim/sample_df_exp_1_dim)*sample_df_unexp_1_dim)
        
        if sample_df_unexp_1_dim/sample_df_unexp_0_dim > sample_df_exp_1_dim/sample_df_exp_0_dim
            new_df_unexp_1 = sample_df_unexp_1[sample(1:sample_df_unexp_1_dim,dim_sampleA , replace=false),:]
            dfd_sample = vcat(sample_df_exp,new_df_unexp_1,sample_df_unexp_0)
        else
            new_df_unexp_0 = sample_df_unexp_0[sample(1:sample_df_unexp_0_dim, dim_sampleB, replace=false ),:]
            dfd_sample = vcat(sample_df_exp,sample_df_unexp_1,new_df_unexp_0)
        end
    end 
    rows2remove = setdiff(dfd[dfd[:isO].==false, :panid],dfd_sample[:panid])
    #dfd[findin(dfd[:panid],rows2remove),:whyO]="NoMatch"
    dfd[findin(dfd[:panid],rows2remove),:isO]=true 
    return dfd[dfd[:isO].==false, : ]  #[setdiff(names(dfd),[:isO,:whyO])] 
end

dfd = MatchMe(dfd,cfg)

lowercase!(dfd)
cfg=lowercase(cfg)

#SAVE FILE
#writetable(root*"/matched_dfd.csv", dfd)
#dfd = readtable(root*"/matched_dfd.csv",header=true);

######################################
#------------MODEL OBJECTS-----------#  [:fea_or_dis_trps_shr_dpp_p1,:fea_or_dis_trps_shr_dpp_p2,:fea_or_dis_trps_shr_dpp_p3,:fea_or_dis_trps_shr_dpp_p4]
######################################

iocc = Dict(:modelName=>:occ, :raneff=>cfg[:random_campaigns], :y_var=>:trps_pos_p1, :logvar=>:LOG_trps_pre_p1, :logvarOrig=>:trps_pre_p1 )
idolocc = Dict(:modelName=>:dolocc, :raneff=>cfg[:random_campaigns], :y_var=>:dol_per_trip_pos_p1, :logvar=>:LOG_dol_per_trip_pre_p1, :logvarOrig=>:dol_per_trip_pre_p1)
ipen = Dict(:modelName=>:pen, :raneff=>cfg[:random_campaigns], :y_var=>:buyer_pos_p1, :logvar=>:LOG_buyer_pre_p1, :logvarOrig=>:buyer_pre_p1 )

custom_vars = [:iso,:whyo,:data_nb_ne_b, :data_b_e_nb ,:pen_reduction,:occ_reduction,:dolocc_reduction]  
for m in [iocc,idolocc,ipen] dfd[m[:logvar]]=log(Array(dfd[m[:logvarOrig]]+1)) end

function genExcludeVars!(iocc::Dict,idolocc::Dict,ipen::Dict)  
    iocc[:exclude_vars] = vcat( custom_vars,  
                                [ :buyer_pos_p1, iocc[:logvarOrig]
                                  ,idolocc[:y_var],idolocc[:logvar],idolocc[:logvarOrig] 
                                 ,ipen[:y_var],ipen[:logvar],ipen[:logvarOrig] 
                               ]
                              )
    idolocc[:exclude_vars] = vcat( custom_vars,  
                                   [ :buyer_pos_p1, idolocc[:logvarOrig]
                                           ,iocc[:y_var],iocc[:logvar],iocc[:logvarOrig] 
                                           ,ipen[:y_var],ipen[:logvar],ipen[:logvarOrig] 
                                   ]
                                 )    
    
    ipen[:exclude_vars] = vcat( custom_vars,
                                [ ipen[:logvarOrig]
                                  ,iocc[:y_var],iocc[:logvar],iocc[:logvarOrig] 
                                  ,idolocc[:y_var],idolocc[:logvar],idolocc[:logvarOrig] 
                                ]
                                 )        
end
genExcludeVars!(iocc,idolocc,ipen)


#function expandM(di::Dict)
#    d2=deepcopy(di)
#    if d2[:modelName]==:occ  
#        d2[:dist] = Poisson()
#        d2[:lnk] = LogLink()
#        d2[:Buyer_Pos_P1_is1] = true
#    end
#    if d2[:modelName]==:dolocc  
#        d2[:dist] = Gamma()
#        d2[:lnk] = LogLink()
#        d2[:Buyer_Pos_P1_is1] = true
#    end
#    if d2[:modelName]==:pen
#        d2[:dist] = Bernoulli()
#        d2[:lnk] = LogitLink()
#        d2[:Buyer_Pos_P1_is1] = false
#    end
#    return d2
#end
#expandM(t)






function featureSelection(dfd::DataFrame, m::Dict)
    function rmVars(v::DataArray{Any}) rmVars(convert(Array{Symbol},v)) end
    function rmVars(v::Array{Any}) rmVars(convert(Array{Symbol},v)) end
    function rmVars(v::Array{Symbol})
        #v=setdiff(v,[:group])
        return setdiff(vars,v)  
    end        
    vars=setdiff(names(dfd),   vcat(m[:exclude_vars],m[:y_var],:panid,cfg[:random_demos],cfg[:random_campaigns],cfg[:scoring_vars]) )
    upperModName = uppercase( string(  m[:modelName]  )  )
    println( upperModName*" : SingleValue") #SingleValue
    removed_SingleLevelVars=FS_singleLevel(dfd,vars)
    vars = rmVars(removed_SingleLevelVars)
    println(upperModName*" : Singularity : "*string(genF(m[:y_var],vars))) # Singularity
    singularity_x = checksingularity(genF(m[:y_var],vars), dfd)
    vars = rmVars(singularity_x)
    println(upperModName*" : PVals") #PVals
    f=genF(m[:y_var],vars)
    g1 = glm( f, dfd[convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end]))]  , m[:dist], m[:lnk] )
    sdf = coefDF(g1,true)
    #g1_x= sdf[sdf[:pval].>0.7,:vars]
    g1_x= sdf[sdf[:pval].>0.7,:parameter]
    #vars = rmVars(g1_x)
    vars = rmVars(convert(Array{Any},g1_x))
    #function chkSigns(m::Dict, vars::Array{Symbol}, dfd::DataFrame, cfg::OrderedDict)  # Pvalue & Signs
    #    vars=unique(vcat(vars,[:group]))
    #    f=genF(m[:y_var],vars)
    #    g = glm( f, dfd[convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end]))]  , m[:dist], m[:lnk] )
    #    sdf = coefDF(g1)
    #    neutralvars = setdiff(vars,vcat(cfg[:negativevars],cfg[:positivevars])) 
    #    neg=intersect(cfg[:negativevars],sdf[sdf[:coef].<0,:vars])
    #    pos=intersect(cfg[:positivevars],sdf[sdf[:coef].>0,:vars])
    #    varstokeep = intersect(vcat(neutralvars, pos,neg) ,  sdf[sdf[:pval].<cfg[:pvalue_lvl] ,:vars] )
    #    varstokeep =  convert(Array{Symbol},varstokeep)
    #    return g, varstokeep
    #end
    function chkSigns(m::Dict, vars::Array{Symbol}, dfd::DataFrame, cfg::OrderedDict)  # Pvalue & Signs
        vars=unique(vcat(vars,[:group]))
        f=genF(m[:y_var],vars)
        g = glm( f, dfd[convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end]))]  , m[:dist], m[:lnk] )
        sdf = coefDF(g1, true)
        neutralvars = setdiff(vars,vcat(cfg[:negativevars],cfg[:positivevars])) 
        neg=intersect(cfg[:negativevars],sdf[sdf[:coef].<0,:parameter])
        pos=intersect(cfg[:positivevars],sdf[sdf[:coef].>0,:parameter])
        varstokeep = intersect(vcat(neutralvars, pos,neg) ,  sdf[sdf[:pval].<cfg[:pvalue_lvl] ,:parameter] )
        varstokeep =  convert(Array{Symbol},varstokeep)
        return g, varstokeep
    end
    
    println(upperModName*" : SIGN Check 1") 
    (g3, initialvars) = chkSigns(m, vars, dfd, cfg)
    println(upperModName*" : SIGN Check 2") 
    (g4, vars_2) = chkSigns(m, initialvars, dfd, cfg)
    function getCorrVars(m::RegressionModel, dfd::DataFrame, vars_2::Array{Symbol})
        sdf = coefDF(m, true)
        vars_2 = convert(Array{Symbol}, setdiff(vars_2, filter(x-> typeof(dfd[x]) in [PooledDataArray{String,UInt8,1}], vars_2)) )
        rm_lst=Symbol[]
        if (length(vars_2) > 1) & (   length(getColswithType("num", dfd, vars_2 ) ) > 1  )
            stackdf = corDFD(dfd,vars_2)
            stackdf[:variable_pval] = [ sdf[sdf[:parameter].==c,:pval][1]   for c in stackdf[:variable]]
            stackdf[:vars_pval] = [ sdf[sdf[:parameter].==c,:pval][1]   for c in stackdf[:vars]] 
            stackdf[:most_Sig] = map((x,y) -> x < y ? "variable" : "vars" ,stackdf[:variable_pval],stackdf[:vars_pval])
     
            for row in eachrow(stackdf[(stackdf[:value].> 0.8) | (stackdf[:value].<-0.8),:])
                if row[:vars] == "group"
                    push!(rm_lst,row[:variable])
                elseif row[:variable] == "group"
                    push!(rm_lst,row[:vars])
                else
                    row[:most_Sig] == "variable" ? push!(rm_lst,row[:vars]) : push!(rm_lst,row[:variable])
                end
            end    
        end
        return rm_lst
    end
    println(upperModName*" : Correlation") 
    corrvars_x = getCorrVars(g4, dfd,vars_2  )
    vars_2 = setdiff(vars_2,corrvars_x)    
    (g5, finalvars) =  chkSigns(m, convert(Array{Symbol},vars_2), dfd, cfg)
    
    #VIF is only valid on multiple cols
    if length(finalvars) > 2     
        println( upperModName *" : Z & Vif") #Z & Vif
        f=genF(m[:y_var],finalvars)
        g2 = glm( f, dfd[convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end]))]  , m[:dist], m[:lnk] )
        sdf=vifDF2(g2)
        z = sdf[abs(sdf[:zval]).<1.96,:parameter]
        v = sdf[ !DataArrays.isna(sdf[:vif])&(sdf[:vif].>15),:parameter]
        g2_x =intersect(z,v)
        finalvars = setdiff(finalvars,g2_x)
        println("vif_vars: ",g2_x)
    end
    finalvars = setdiff(finalvars,[:group])# reorder for group
    finalvars = convert(Array{Symbol},vcat(finalvars,[:group]) )
    return finalvars
end


factor_cols=vcat( [ cfg[:proscore], :group, :panid], cfg[:allrandoms] )
for c in setdiff(factor_cols,[:panid, cfg[:proscore]]) #cfg[:random_campaigns]
    if !( typeof(dfd[c]) in [  DataArray{String,1}  ]) 
        println("converting to Strings : ", c," of type : ",typeof(dfd[c]))
        dfd[c] = map(x->string(x),dfd[c])
        dfd[c] = convert(Array{String},dfd[c]) 
    end
end
poolit!(dfd,factor_cols)


iocc[:finalvars] = featureSelection(dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:], expandM(iocc))
idolocc[:finalvars]   = featureSelection(dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:], expandM(idolocc))
ipen[:finalvars]  = featureSelection(dfd[(dfd[:iso].==false) ,:], expandM(ipen))

#m=expandM(ipen)
#glm(genF(m[:y_var],m[:finalvars]), dfd[vcat(m[:y_var],m[:finalvars])], m[:dist], m[:lnk])



# --- Removing outlliers - to ensure overall campaign uplift
function getOutliers(r::DataFrame, dfd::DataFrame, m::Dict, segment::Symbol,pct::Int64) # :data_nb_ne_b, :data_b_e_nb, :pen_reduction, :occ_reduction, :dolocc_reduction
    if m[:Buyer_Pos_P1_is1]
        println("pre=1")
        odf = dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1)&(dfd[segment].==true),:panid]
    else
        println("pre!=1")
        odf = dfd[(dfd[:iso].==false)&(dfd[segment].==true),:panid]
    end
    pctnum = Integer(round(length(odf) / 100) * pct)
    return sort(r[findin(r[:panid],odf) ,:], cols=[order(:resids, rev = true)] )[1:pctnum,:panid]
end

function genGLM(dfd::DataFrame, m::Dict)
    f=genF(m[:y_var],m[:finalvars])
    println("GLM for ",m[:modelName]," ::: ",f)
    dfd_tmp = m[:modelName] in [:occ, :dolocc] ? dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:] : dfd[(dfd[:iso].==false),:] 
    g = glm(f, dfd_tmp[vcat(m[:y_var],m[:finalvars])], m[:dist], m[:lnk])
    println("GLM Residuals for ",m[:modelName])
    r = DataFrame(panid=dfd_tmp[:panid], resids=StatLib.xResiduals(g))
    return r, g 
end

function genlists(dfd::DataFrame, iocc::Dict,idolocc::Dict,ipen::Dict)   
    r1, g1 = genGLM( dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:], expandM(iocc))
    r2, g2 = genGLM( dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:], expandM(idolocc))
    r3, g3 = genGLM( dfd[(dfd[:iso].==false),:], expandM(ipen))
    rlist = [r1,r2,r3]
    glist = [g1,g2,g3]
    return rlist, glist
end
rlist, glist = genlists(dfd,iocc,idolocc,ipen) 

for i in 1:5
    mlist = [expandM(iocc),expandM(idolocc),expandM(ipen)]
    #mliftβ = [ g.sdf[g.sdf[:vars].==:group,:coef][1] for g in glist]
    mliftβ = [ sdf[sdf[:parameter].==:group,:coef][1] for sdf in [coefDF(m, true) for m in glist]]
    minidx = find(x->x==minimum(mliftβ),mliftβ)[1]
    if  mliftβ[minidx] < 0.0
        #mx=mlist[minidx]
        println("WRONG : ",mliftβ," for ",mlist[minidx][:modelName])
        panids = getOutliers(rlist[minidx], dfd, mlist[minidx], :data_nb_ne_b,5)
        dfd[findin(dfd[:panid],panids) ,:iso] = true
        rlist, glist = genlists(dfd,iocc,idolocc,ipen) 
    else
        println("All good : ",mliftβ)
        break
    end
end



# ---- SAVE RESULTS -----------
delete!(iocc, :exclude_vars)
delete!(idolocc, :exclude_vars)
delete!(ipen, :exclude_vars)

modelsDict = Dict()
modelsDict[:iocc] = iocc
modelsDict[:idolocc] = idolocc
modelsDict[:ipen] = ipen


function unpool(dfd::DataFrame)
    v_factors = Symbol[]
    for c in names(dfd)
        if typeof(dfd[c]) in [ PooledDataArray{String,UInt8,1} ]
            println("         Converting : ", c )
            push!(v_factors,c)
            dfd[c] = Array(dfd[c])
        end
    end
    return v_factors
end

modelsDict[:factors] = unpool(dfd) 

jmod_fname = root*"/dfd_model.json"
mod_fname = root*"/dfd_model.csv" 

#writetable(root*"/dfd_postOutliers.csv", dfd)
cols = vcat([:panid, :iso, :group, :buyer_pos_p1],cfg[:random_campaigns])
cols = vcat(cols,iocc[:finalvars],[iocc[:y_var], iocc[:logvar]])
cols = vcat(cols,idolocc[:finalvars],[idolocc[:y_var], idolocc[:logvar]])
cols = unique(vcat(cols,ipen[:finalvars],[ipen[:y_var], ipen[:logvar]]) )
# add required cold for Scoring agregation
cols = unique(vcat(cols,[:buyer_pre_p1, :buyer_pos_p1, :trps_pos_p1,:trps_pre_p1,:dol_per_trip_pre_p1,:dol_per_trip_pos_p1, :prd_1_net_pr_pre, :prd_1_net_pr_pos]))
writetable(mod_fname, dfd[ (dfd[:iso].==false) ,cols])

 
#using JSON

function saveModels(d::Dict,fname::String)
    j = JSON.json([ d])
    open(fname,"w") do f
        write(f,j)
    end
end
saveModels(modelsDict, jmod_fname)

