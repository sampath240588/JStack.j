




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


# ---------------------------------

drop table Jennieo6.Kantar_FSP_Jennieo6_2_931_dk;
Create External Table if not exists Jennieo6.Kantar_FSP_Jennieo6_2_931_dk
(
VENUE_DIM_KEY BIGINT,
TM_DIM_KEY_DAY BIGINT,
date1 STRING,
Trans_TIME STRING,
Trans_NUM DECIMAL(20,0),
STORE_NUM BIGINT,
CARD_NUM BIGINT,
UPC BIGINT,
ndc_upc_ind string,
QUANTITY double ,
WEIGHT double,
item_list_price double,
NET_PRICE double ,
CARD_DISCOUNT double ,
ad_discount double,
other_discount double,
ITEM_TYPE   string ,  
TENDER_TYPE   int ,
ad_event     string,           
ad_version bigint,  
TM_DIM_KEY_WEEK BIGINT,
OPERATING_COMPANY STRING,
item_dim_key bigint,
UPC10 STRING,
UNITS INT,
CENTS INT,
BASELINE_UNITS DOUBLE ,
BASELINE_CENTS DOUBLE ,
FEATURE INT,
DISPLAY INT,
TOTL_PRC_REDUC INT,
registration_request_id BIGINT,
product_id_Jennieo6 INT,
household_id BIGINT,
HTML5 int,
Static int,
Native_creative int,
Video int,
Content int,
Contextual int,
Direct int,
Prospecting int,
Retargeting int,
Behavorial int,
Native int,
Predictive int,
3rdparty int,
PMP int,
Buzzfeed int,
Hulu int,
PopSugarUS int,
RachaelRayMag int,
Shape int,
Blank int,
EatingWell int,
MeredithCorporation int,
RealSimple int,
TURNDSP int,
CookingLight int,
dataxu int,
Allrecipes int,
Amazon int,
MyFitnessPal int,
WomensHealth int,
Yummly int,
YouTube int,
gross_imps bigint)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|' LOCATION '/mapr/mapr04p/analytics0001/analytic_users/mddak/jennie/Kantar_FSP_Jennieo6_2_931_dk';

Insert Overwrite Table Jennieo6.Kantar_FSP_Jennieo6_2_931_dk  
select
FSP.VENUE_DIM_KEY,
FSP.TM_DIM_KEY_DAY,
FROM_UNIXTIME((unix_timestamp(cast(FSP.Trans_DATE as string),'YYYY-MM-DD'))) as date1,
FSP.Trans_TIME,
FSP.Trans_NUM,
FSP.STORE_NUM,
FSP.CARD_NUM,
FSP.UPC,
FSP.ndc_upc_ind,
FSP.QUANTITY,
FSP.WEIGHT,
FSP.item_list_price,
FSP.NET_PRICE,
FSP.CARD_DISCOUNT,
FSP.ad_discount,
FSP.other_discount,
FSP.ITEM_TYPE,  
FSP.TENDER_TYPE,
FSP.ad_event,                
FSP.ad_version ,
FSP.TM_DIM_KEY_WEEK,
FSP.OPERATING_COMPANY,
FSP.item_dim_key,
FSP.UPC10,
FSP.UNITS,
FSP.CENTS,
FSP.BASELINE_UNITS,
FSP.BASELINE_CENTS,
FSP.FEATURE,
FSP.DISPLAY,
FSP.TOTL_PRC_REDUC,
FSP.registration_request_id,
FSP.Product_id_Jennieo6_final,
exp.household_id,
0 as HTML5,
0 as Static,
0 as Native_creative,
0 as Video,
0 as Content,
0 as Contextual,
0 as Direct,
0 as Prospecting,
0 as Retargeting,
0 as Behavorial,
0 as Native,
0 as Predictive,
0 as 3rdparty,
0 as PMP,
0 as Buzzfeed,
0 as Hulu,
0 as PopSugarUS,
0 as RachaelRayMag,
0 as Shape,
0 as Blank,
0 as EatingWell,
0 as MeredithCorporation,
0 as RealSimple,
0 as TURNDSP,
0 as CookingLight,
0 as dataxu,
0 as Allrecipes,
0 as Amazon,
0 as MyFitnessPal,
0 as WomensHealth,
0 as Yummly,
0 as YouTube,
0 as gross_imps
from Jennieo6.Kantar_FSP_POS_Jennieo6_PRDMAP_931_final FSP
join WH_SUPPLMENTAL.HOUSEHOLD_DEPENDENT_ID_MAP exp on
    cast(FSP.CARD_NUM as BIGINT) = cast(exp.dependent_id as BIGINT) 
    AND FSP.Operating_company= exp.retailer
where (FSP.card_num is not NULL) 
       and (trim(FSP.operating_company) not in ('FREDS')) 
       and (FSP.UPC10<>'2265570029') 
       and (FSP.UPC10<>'2265570051') 
       and (FSP.UPC10<>'2265570140')
       and exp.household_type = 'EXPERIAN'  
       and exp.dependent_type = 'FSP_CARD'  
       and exp.current_state='MATCHED'
order by household_id, date1, Trans_TIME;

    
    
set hive.cli.print.header=true; select date1 from Jennieo6.Kantar_FSP_Jennieo6_2_931_dk limit 10;
hive -e 'set hive.cli.print.header=true; select * from Jennieo6.Kantar_FSP_Jennieo6_2_931_dk limit 10' | sed 's/[\t]/,/g' > /mapr/mapr04p/analytics0001/analytic_users/mddak/lift_product_example10.csv

drop table if exists jennie.jennie6_daily_breaks_ru;
CREATE EXTERNAL TABLE IF NOT EXISTS jennie.jennie6_daily_breaks_ru(
VENUE_DIM_KEY BIGINT,
TM_DIM_KEY_DAY BIGINT,
date1 STRING,
Trans_TIME STRING,
Trans_NUM DECIMAL(20,0),
STORE_NUM BIGINT,
CARD_NUM BIGINT,
UPC BIGINT,
ndc_upc_ind string,
QUANTITY double ,
WEIGHT double,
item_list_price double,
NET_PRICE double ,
CARD_DISCOUNT double ,
ad_discount double,
other_discount double,
ITEM_TYPE   string ,  
TENDER_TYPE   int ,
ad_event     string,           
ad_version bigint,  
TM_DIM_KEY_WEEK BIGINT,
OPERATING_COMPANY STRING,
item_dim_key bigint,
UPC10 STRING,
UNITS INT,
CENTS INT,
BASELINE_UNITS DOUBLE ,
BASELINE_CENTS DOUBLE ,
FEATURE INT,
DISPLAY INT,
TOTL_PRC_REDUC INT,
registration_request_id BIGINT,
product_id_Jennieo6 INT,
household_id BIGINT,
HTML5 int,
Static int,
Native_creative int,
Video int,
Content int,
Contextual int,
Direct int,
Prospecting int,
Retargeting int,
Behavorial int,
Native int,
Predictive int,
3rdparty int,
PMP int,
Buzzfeed int,
Hulu int,
PopSugarUS int,
RachaelRayMag int,
Shape int,
Blank int,
EatingWell int,
MeredithCorporation int,
RealSimple int,
TURNDSP int,
CookingLight int,
dataxu int,
Allrecipes int,
Amazon int,
MyFitnessPal int,
WomensHealth int,
Yummly int,
YouTube int,
gross_imps bigint)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','  LOCATION '/mapr/mapr04p/analytics0001/analytic_users/mddak/jennie/jennie6_daily_breaks_ru';

Insert Overwrite Table jennie.jennie6_daily_breaks_ru
select
'0' as VENUE_DIM_KEY,
'0' as TM_DIM_KEY_DAY,
FROM_UNIXTIME((unix_timestamp(cast(a.date1 as string),"yyyyMMdd"))) as date1,
'na' as Trans_TIME,
'0' as Trans_NUM,
'0' as STORE_NUM,
'0' as CARD_NUM,
'0' as UPC,
'na' as ndc_upc_ind,
'0' as QUANTITY,
'0' as WEIGHT,
'0' as item_list_price,
'0' as NET_PRICE,
'0' as CARD_DISCOUNT,
'0' as ad_discount,
'0' as other_discount,
'na' as ITEM_TYPE,
'0' as TENDER_TYPE,
'na' as ad_event,
'0' as ad_version,
'0' as TM_DIM_KEY_WEEK,
max(b.retailer) as OPERATING_COMPANY,
'0' as item_dim_key,
'na' as UPC10,
'0' as UNITS,
'0' as CENTS,
'0' as BASELINE_UNITS,
'0' as BASELINE_CENTS,
'0' as FEATURE,
'0' as DISPLAY,
'0' as TOTL_PRC_REDUC,
'0' as registration_request_id,
'0' as product_id_Jennieo6,
b.household_id,
sum(a.HTML5) as HTML5,
sum(a.Static) as Static,
sum(a.Native_creative) as Native_creative,
sum(a.Video) as Video,
sum(a.Content) as Content,
sum(a.Contextual) as Contextual,
sum(a.Direct) as Direct,
sum(a.Prospecting) as Prospecting,
sum(a.Retargeting) as Retargeting,
sum(a.Behavorial) as Behavorial,
sum(a.Native) as Native,
sum(a.Predictive) as Predictive,
sum(a.3rdparty) as 3rdparty,
sum(a.PMP) as PMP,
sum(a.Buzzfeed) as Buzzfeed,
sum(a.Hulu) as Hulu,
sum(a.PopSugarUS) as PopSugarUS,
sum(a.RachaelRayMag) as RachaelRayMag,
sum(a.Shape) as Shape,
sum(a.Blank) as Blank,
sum(a.EatingWell) as EatingWell,
sum(a.MeredithCorporation) as MeredithCorporation,
sum(a.RealSimple) as RealSimple,
sum(a.TURNDSP) as TURNDSP,
sum(a.CookingLight) as CookingLight,
sum(a.dataxu) as dataxu,
sum(a.Allrecipes) as Allrecipes,
sum(a.Amazon) as Amazon,
sum(a.MyFitnessPal) as MyFitnessPal,
sum(a.WomensHealth) as WomensHealth,
sum(a.Yummly) as Yummly,
sum(a.YouTube) as YouTube,
sum(a.gross_imps) as gross_imps
from jennie.jennie6_daily_breaks a
inner JOIN WH_SUPPLMENTAL.HOUSEHOLD_DEPENDENT_ID_MAP b on a.lr_id = b.dependent_id 
where b.dependent_type = 'LVRAMP_ID' and b.household_type = 'EXPERIAN' and b.retailer = 'COMSCORE' and b.data_supplier = 'EXPERIAN' and b.current_state = 'MATCHED'
group by household_id, date1
order by household_id, date1;

set hive.cli.print.header=true; select date1 from jennie.jennie6_daily_breaks_ru limit 10;

-- union the pos and exposure file into one table
drop table jennie.jennie_rf_pos_exposure;
Create External Table if not exists jennie.jennie_rf_pos_exposure
(
VENUE_DIM_KEY BIGINT,
TM_DIM_KEY_DAY BIGINT,
date1 STRING,
Trans_TIME STRING,
Trans_NUM DECIMAL(20,0),
STORE_NUM BIGINT,
CARD_NUM BIGINT,
UPC BIGINT,
ndc_upc_ind string,
QUANTITY double ,
WEIGHT double,
item_list_price double,
NET_PRICE double ,
CARD_DISCOUNT double ,
ad_discount double,
other_discount double,
ITEM_TYPE   string ,  
TENDER_TYPE   int ,
ad_event     string,           
ad_version bigint,  
TM_DIM_KEY_WEEK BIGINT,
OPERATING_COMPANY STRING,
item_dim_key bigint,
UPC10 STRING,
UNITS INT,
CENTS INT,
BASELINE_UNITS DOUBLE ,
BASELINE_CENTS DOUBLE ,
FEATURE INT,
DISPLAY INT,
TOTL_PRC_REDUC INT,
registration_request_id BIGINT,
product_id_Jennieo6 INT,
household_id BIGINT,
HTML5 int,
Static int,
Native_creative int,
Video int,
Content int,
Contextual int,
Direct int,
Prospecting int,
Retargeting int,
Behavorial int,
Native int,
Predictive int,
3rdparty int,
PMP int,
Buzzfeed int,
Hulu int,
PopSugarUS int,
RachaelRayMag int,
Shape int,
Blank int,
EatingWell int,
MeredithCorporation int,
RealSimple int,
TURNDSP int,
CookingLight int,
dataxu int,
Allrecipes int,
Amazon int,
MyFitnessPal int,
WomensHealth int,
Yummly int,
YouTube int,
gross_imps bigint)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|' LOCATION '/mapr/mapr04p/analytics0001/analytic_users/mddak/jennie/jennie_rf_pos_exposure';

Insert Overwrite Table jennie.jennie_rf_pos_exposure  
select * from (
select
a.VENUE_DIM_KEY,
a.TM_DIM_KEY_DAY,
a.date1,
a.Trans_TIME,
a.Trans_NUM,
a.STORE_NUM,
a.CARD_NUM,
a.UPC,
a.ndc_upc_ind,
a.QUANTITY,
a.WEIGHT,
a.item_list_price,
a.NET_PRICE,
a.CARD_DISCOUNT,
a.ad_discount,
a.other_discount,
a.ITEM_TYPE,  
a.TENDER_TYPE,
a.ad_event,                
a.ad_version ,
a.TM_DIM_KEY_WEEK,
a.OPERATING_COMPANY,
a.item_dim_key,
a.UPC10,
a.UNITS,
a.CENTS,
a.BASELINE_UNITS,
a.BASELINE_CENTS,
a.FEATURE,
a.DISPLAY,
a.TOTL_PRC_REDUC,
a.registration_request_id,
a.Product_id_Jennieo6,
a.household_id,
a.HTML5,
a.Static,
a.Native_creative,
a.Video,
a.Content,
a.Contextual,
a.Direct,
a.Prospecting,
a.Retargeting,
a.Behavorial,
a.Native,
a.Predictive,
a.3rdparty,
a.PMP,
a.Buzzfeed,
a.Hulu,
a.PopSugarUS,
a.RachaelRayMag,
a.Shape,
a.Blank,
a.EatingWell,
a.MeredithCorporation,
a.RealSimple,
a.TURNDSP,
a.CookingLight,
a.dataxu,
a.Allrecipes,
a.Amazon,
a.MyFitnessPal,
a.WomensHealth,
a.Yummly,
a.YouTube,
a.gross_imps
from Jennieo6.Kantar_FSP_Jennieo6_2_931_dk a
union all
select
b.VENUE_DIM_KEY,
b.TM_DIM_KEY_DAY,
b.date1,
b.Trans_TIME,
b.Trans_NUM,
b.STORE_NUM,
b.CARD_NUM,
b.UPC,
b.ndc_upc_ind,
b.QUANTITY,
b.WEIGHT,
b.item_list_price,
b.NET_PRICE,
b.CARD_DISCOUNT,
b.ad_discount,
b.other_discount,
b.ITEM_TYPE,  
b.TENDER_TYPE,
b.ad_event,                
b.ad_version ,
b.TM_DIM_KEY_WEEK,
b.OPERATING_COMPANY,
b.item_dim_key,
b.UPC10,
b.UNITS,
b.CENTS,
b.BASELINE_UNITS,
b.BASELINE_CENTS,
b.FEATURE,
b.DISPLAY,
b.TOTL_PRC_REDUC,
b.registration_request_id,
b.Product_id_Jennieo6,
b.household_id,
b.HTML5,
b.Static,
b.Native_creative,
b.Video,
b.Content,
b.Contextual,
b.Direct,
b.Prospecting,
b.Retargeting,
b.Behavorial,
b.Native,
b.Predictive,
b.3rdparty,
b.PMP,
b.Buzzfeed,
b.Hulu,
b.PopSugarUS,
b.RachaelRayMag,
b.Shape,
b.Blank,
b.EatingWell,
b.MeredithCorporation,
b.RealSimple,
b.TURNDSP,
b.CookingLight,
b.dataxu,
b.Allrecipes,
b.Amazon,
b.MyFitnessPal,
b.WomensHealth,
b.Yummly,
b.YouTube,
b.gross_imps
from jennie.jennie6_daily_breaks_ru b 
) unioned;

set hive.cli.print.header=true; select * from jennie.jennie_rf_pos_exposure limit 10;


-- prepare table for analysis
drop table jennie.jennie_rf_pos_exposure2;
Create External Table if not exists jennie.jennie_rf_pos_exposure2
(
VENUE_DIM_KEY BIGINT,
TM_DIM_KEY_DAY BIGINT,
date1 STRING,
Trans_TIME STRING,
Trans_NUM DECIMAL(20,0),
STORE_NUM BIGINT,
CARD_NUM BIGINT,
UPC BIGINT,
ndc_upc_ind string,
QUANTITY double ,
WEIGHT double,
item_list_price double,
NET_PRICE double ,
CARD_DISCOUNT double ,
ad_discount double,
other_discount double,
ITEM_TYPE   string ,  
TENDER_TYPE   int ,
ad_event     string,           
ad_version bigint,  
TM_DIM_KEY_WEEK BIGINT,
OPERATING_COMPANY STRING,
item_dim_key bigint,
UPC10 STRING,
UNITS INT,
CENTS INT,
BASELINE_UNITS DOUBLE ,
BASELINE_CENTS DOUBLE ,
FEATURE INT,
DISPLAY INT,
TOTL_PRC_REDUC INT,
registration_request_id BIGINT,
product_id_Jennieo6 INT,
household_id BIGINT,
HTML5 int,
Static int,
Native_creative int,
Video int,
Content int,
Contextual int,
Direct int,
Prospecting int,
Retargeting int,
Behavorial int,
Native int,
Predictive int,
3rdparty int,
PMP int,
Buzzfeed int,
Hulu int,
PopSugarUS int,
RachaelRayMag int,
Shape int,
Blank int,
EatingWell int,
MeredithCorporation int,
RealSimple int,
TURNDSP int,
CookingLight int,
dataxu int,
Allrecipes int,
Amazon int,
MyFitnessPal int,
WomensHealth int,
Yummly int,
YouTube int,
gross_imps bigint)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|' LOCATION '/mapr/mapr04p/analytics0001/analytic_users/mddak/jennie/jennie_rf_pos_exposure2';

Insert Overwrite Table jennie.jennie_rf_pos_exposure2
select
max(a.VENUE_DIM_KEY) as VENUE_DIM_KEY,
max(a.TM_DIM_KEY_DAY) as TM_DIM_KEY_DAY,
a.date1,
max(a.Trans_TIME) as Trans_TIME,
max(a.Trans_NUM) as Trans_NUM,
max(a.STORE_NUM) as STORE_NUM,
max(a.CARD_NUM) as CARD_NUM,
max(a.UPC) as UPC,
max(a.ndc_upc_ind) as ndc_upc_ind,
sum(a.QUANTITY) as QUANTITY,
max(a.WEIGHT) as WEIGHT,
max(a.item_list_price) as item_list_price,
max(a.NET_PRICE) as NET_PRICE,
max(a.CARD_DISCOUNT) as CARD_DISCOUNT,
max(a.ad_discount) as ad_discount,
max(a.other_discount) as other_discount,
max(a.ITEM_TYPE) as ITEM_TYPE,
max(a.TENDER_TYPE) as TENDER_TYPE,
max(a.ad_event) as ad_event,
max(a.ad_version) as ad_version,
max(a.TM_DIM_KEY_WEEK) as TM_DIM_KEY_WEEK,
max(a.OPERATING_COMPANY) as OPERATING_COMPANY,
max(a.item_dim_key) as item_dim_key,
max(a.UPC10) as UPC10,
sum(a.UNITS) as UNITS,
sum(a.CENTS) as CENTS,
sum(a.BASELINE_UNITS) as BASELINE_UNITS,
sum(a.BASELINE_CENTS) as BASELINE_CENTS,
max(a.FEATURE) as FEATURE,
max(a.DISPLAY) as DISPLAY,
sum(a.TOTL_PRC_REDUC) as TOTL_PRC_REDUC,
max(a.registration_request_id) as registration_request_id,
max(a.Product_id_Jennieo6) as product_id_Jennieo6,
a.household_id,
sum(a.HTML5) as HTML5,
sum(a.Static) as Static,
sum(a.Native_creative) as Native_creative,
sum(a.Video) as Video,
sum(a.Content) as Content,
sum(a.Contextual) as Contextual,
sum(a.Direct) as Direct,
sum(a.Prospecting) as Prospecting,
sum(a.Retargeting) as Retargeting,
sum(a.Behavorial) as Behavorial,
sum(a.Native) as Native,
sum(a.Predictive) as Predictive,
sum(a.3rdparty) as 3rdparty,
sum(a.PMP) as PMP,
sum(a.Buzzfeed) as Buzzfeed,
sum(a.Hulu) as Hulu,
sum(a.PopSugarUS) as PopSugarUS,
sum(a.RachaelRayMag) as RachaelRayMag,
sum(a.Shape) as Shape,
sum(a.Blank) as Blank,
sum(a.EatingWell) as EatingWell,
sum(a.MeredithCorporation) as MeredithCorporation,
sum(a.RealSimple) as RealSimple,
sum(a.TURNDSP) as TURNDSP,
sum(a.CookingLight) as CookingLight,
sum(a.dataxu) as dataxu,
sum(a.Allrecipes) as Allrecipes,
sum(a.Amazon) as Amazon,
sum(a.MyFitnessPal) as MyFitnessPal,
sum(a.WomensHealth) as WomensHealth,
sum(a.Yummly) as Yummly,
sum(a.YouTube) as YouTube,
sum(a.gross_imps) as gross_imps
from jennie.jennie_rf_pos_exposure a
group by household_id, date1
order by household_id, date1;

set hive.cli.print.header=true; select * from jennie.jennie_rf_pos_exposure2 limit 10;
                    
                    
===================================================================================
select count(*) from jennie.jennie_rf_pos_exposure where date1 is null;
                    
select quantity, item_type, ndc_upc_ind, '---', html5, static , native_creative , prospecting from  jennie.jennie_rf_pos_exposure where date1 is null;
                    
                    
                    
Jennieo6.Kantar_FSP_Jennieo6_2_931_dk
select count(*) from Jennieo6.Kantar_FSP_Jennieo6_2_931_dk;
select count(*) from Jennieo6.Kantar_FSP_Jennieo6_2_931_dk where date1 is null;                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
