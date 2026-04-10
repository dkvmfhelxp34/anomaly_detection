# BiLSTM Re-Inference Evaluation Summary

## Methodology
- Re-inference executed from model checkpoints in `output/bilstm/ai_models`.
- POT threshold computed by `pot_eval_stable` with scaler from `codex_scr/pot_configs.json` (`data_name/pair` key).
- Metrics: Recall, F1 for `1QC`, `AI POT`, `1QC+AI(POT)` against 2QC target.
- Exclusion rule: original value NaN OR any of 2QC/1QC/AI_POT NaN.
- Qualitative outputs: POT-only plots from `plot_all_features_anomaly_with_flags`.

## Run Summary
- Processed experiments: 36
- Valid train/test rows: 72
- AI usable rows (`recall_ai_pot > recall_1qc` and `f1_ai_pot > f1_1qc`): 6
- Failed experiments: 0

## Per-Experiment Quantitative Table (Train/Test)
top_folder,split,n_eval_rows,recall_1qc,f1_1qc,recall_ai_pot,f1_ai_pot,recall_1qc_plus_ai,f1_1qc_plus_ai,ai_usable
SMB1_AIR_press,train,483752,0.7751876544361043,0.873106309239373,0.13944706046901953,0.22537015408959046,0.9032588931884936,0.9026697106648651,False
SMB1_AIR_press,test,34768,0.580199539524175,0.7336244541484715,0.23944742900997698,0.34456101601325234,0.8058326937835764,0.823852491173009,False
SMB1_AIR_solar,train,482322,0.9025970548862116,0.9487518644640194,0.000535475234270415,0.0010690042225666792,0.9026506024096386,0.9481410653017606,False
SMB1_AIR_solar,test,34769,0.9960159362549801,0.998003992015968,0.0,0.0,0.9960159362549801,0.9966777408637874,False
SMB1_AIR_temp,train,483752,0.8935334624488918,0.9436931818181818,0.020174306003873467,0.03414367659109533,0.9059070367979342,0.8763466042154567,False
SMB1_AIR_temp,test,34768,0.4039806347498655,0.5752585216392186,0.11941904249596558,0.18933901918976545,0.5142549757934374,0.6209808379343943,False
SMB1_AIR_wind_dir,train,478538,0.9866910229645094,0.9921714410671331,8.698677800974252e-05,0.00017373175816539266,0.9866910229645094,0.991520979020979,False
SMB1_AIR_wind_dir,test,34766,0.9907161803713528,0.9953364423717521,0.0,0.0,0.9907161803713528,0.9953364423717521,False
SMB1_AIR_wind_max_speed,train,478859,0.9870525514089871,0.9923852469477177,0.008885503935008886,0.017239963878170922,0.9870525514089871,0.9824797843665768,False
SMB1_AIR_wind_max_speed,test,34770,0.9907651715039578,0.9953611663353215,0.0,0.0,0.9907651715039578,0.9953611663353215,False
SMB1_AIR_wind_speed,train,478536,0.9866910229645094,0.9921714410671331,0.009655532359081419,0.018561872909698993,0.9872999304105776,0.9781109962082041,False
SMB1_AIR_wind_speed,test,34764,0.9907161803713528,0.9953364423717521,0.0,0.0,0.9907161803713528,0.9953364423717521,False
SMB1_SURFACE_current_dir,train,465078,0.6209970948675995,0.7251281851324742,0.006878819247336962,0.013277045535110051,0.6228670651484289,0.7153321061512502,False
SMB1_SURFACE_current_dir,test,7683,0.7647058823529411,0.8666666666666666,0.012605042016806723,0.01662049861495845,0.7647058823529411,0.674074074074074,False
SMB1_SURFACE_current_speed,train,465076,0.6209970948675995,0.7251281851324742,0.2655691722042275,0.25784593437945796,0.7447824489932213,0.5860682660220196,False
SMB1_SURFACE_current_speed,test,7681,0.7637130801687764,0.8660287081339713,0.008438818565400843,0.01587301587301587,0.7637130801687764,0.839907192575406,False
SMB2_AIR_press,train,515019,0.8161548328680299,0.8981566054969693,0.09148871953519958,0.1576876808531391,0.8965139968309062,0.9122948459545062,False
SMB2_AIR_press,test,31437,0.8158808933002482,0.8986061765509702,0.0858560794044665,0.15412026726057906,0.8665012406947891,0.9146149816657936,False
SMB2_AIR_solar,train,515497,0.7295628052539238,0.8436384073914146,0.5045995542699986,0.667445452335965,0.7589975816776519,0.8593574884738506,False
SMB2_AIR_solar,test,33810,0.7103132324738973,0.8305976889774815,0.5194971233752397,0.6824113921836185,0.7347645429362881,0.8456256514008952,False
SMB2_AIR_temp,train,515019,0.8764044943820225,0.9340913512461901,0.006733460430779215,0.012904730438838575,0.8806636109195636,0.918556439329836,False
SMB2_AIR_temp,test,31437,0.7338673787271918,0.8465092402464065,0.013796172674677348,0.026863084922010397,0.7467734757454384,0.8485461441213653,False
SMB2_AIR_wind_dir,train,514865,0.9745373527940708,0.986945410171989,0.00013640703860319192,0.0002719115381129339,0.9745373527940708,0.985357331678275,False
SMB2_AIR_wind_dir,test,31454,0.9981905910735827,0.9987929993964997,0.0006031363088057901,0.0011834319526627217,0.9981905910735827,0.9895366218236172,False
SMB2_AIR_wind_max_speed,train,514890,0.9745616425910784,0.9869580218516388,0.0011356409557554284,0.0022519479349637434,0.9747887707822295,0.983455545371219,False
SMB2_AIR_wind_max_speed,test,33815,0.9992526158445441,0.9995016197358585,0.006975585450921774,0.013735589894530293,0.9992526158445441,0.9952853598014888,False
SMB2_AIR_wind_speed,train,514869,0.9745373527940708,0.986945410171989,0.0010457872959578048,0.002066579810413765,0.9748101668712772,0.9816167212289095,False
SMB2_AIR_wind_speed,test,31459,0.9981905910735827,0.9987929993964997,0.0036188178528347406,0.007159904534606205,0.9981905910735827,0.9954887218045114,False
SMB2_SURFACE_current_dir,train,515604,0.41390526291068785,0.5717413871407253,0.00039923909724994715,0.0007976258899980059,0.4141048824593128,0.571674042373843,False
SMB2_SURFACE_current_dir,test,33381,0.8238841978287093,0.9034391534391534,0.0006031363088057901,0.001205303334672559,0.8238841978287093,0.9033395789705722,False
SMB2_SURFACE_current_speed,train,515602,0.41390526291068785,0.5717413871407253,0.28019539231112467,0.37855460105180494,0.5928348324370024,0.6609067881477408,False
SMB2_SURFACE_current_speed,test,33379,0.8238841978287093,0.9034391534391534,0.05669481302774427,0.10681818181818181,0.8568556493767592,0.9205183585313175,False
SMB3_SURFACE_current_dir,train,196960,0.8452023325675915,0.8297417116422513,0.0024543009169268226,0.004893421284425219,0.8456146551216351,0.8297386596796099,False
SMB3_SURFACE_current_dir,test,34594,0.7726207219878106,0.871727056334303,0.0031645569620253164,0.006296641791044776,0.7748476324425692,0.8721635883905012,False
SMB3_SURFACE_current_speed,train,196960,0.8452023325675915,0.8297417116422513,0.3168993343935913,0.4643468504106909,0.9285700261137617,0.8580941321624269,False
SMB3_SURFACE_current_speed,test,34594,0.7726207219878106,0.871727056334303,0.323722456633849,0.46361728913134703,0.8592358180965776,0.8894685755884495,False
SMB3_WATER_O2Per,train,199659,0.9169936287390663,0.8769438635444367,0.15821964652100357,0.2714402606024115,0.9216010942730644,0.8763123892972352,False
SMB3_WATER_O2Per,test,34601,0.670661749641467,0.8028695812128273,0.0,0.0,0.670661749641467,0.8028695812128273,False
SMB3_WATER_O2ppm,train,199942,0.8541629171016162,0.8421584980657985,0.1614952665490803,0.27618153557303127,0.9223390086749937,0.8767364675289127,False
SMB3_WATER_O2ppm,test,34601,0.670661749641467,0.8028695812128273,0.0,0.0,0.670661749641467,0.8028695812128273,False
SMB3_WATER_Salinity,train,199942,0.5352714477211796,0.6585967469977101,0.00045480658751436234,0.0009089604363010093,0.5354749138261202,0.6586821939387694,False
SMB3_WATER_Salinity,test,34601,0.8683373110580748,0.929529486906536,0.0,0.0,0.8683373110580748,0.929529486906536,False
SMB3_WATER_chl-a,train,199942,0.7678545911805893,0.8115320431016907,0.8927581378068646,0.5222694063163525,1.0,0.5653009790300172,False
SMB3_WATER_chl-a,test,34601,0.612670784203631,0.7598212731387455,0.7864495601721879,0.4469737261993405,1.0,0.5358271072556787,False
SMB3_WATER_pH,train,199942,0.5033666805770164,0.6334225111235692,0.07445044719894432,0.13855276276433856,0.5679708559376515,0.6867078034977669,False
SMB3_WATER_pH,test,34601,1.0,1.0,0.0,0.0,1.0,1.0,False
SMB3_WATER_temp,train,199942,0.9699270610432341,0.8808784962751675,0.002669059616667817,0.00532281007663011,0.9699960884471135,0.8808307476938185,False
SMB3_WATER_temp,test,34601,0.9898699727850016,0.994909201428463,0.0,0.0,0.9898699727850016,0.994909201428463,False
SMB3_WATER_turbidity,train,199942,0.5280318710912287,0.6571021035576297,0.2162204268767666,0.3519736246218503,0.6918052124368476,0.7759478266296377,False
SMB3_WATER_turbidity,test,34601,0.5281376783143568,0.6912174024750567,0.5128909828556472,0.6731074756322727,0.9345635388038215,0.9606834157333602,False
SMB4_SURFACE_current_dir,train,203989,0.29696814852665643,0.45502250971692754,0.018188675572426212,0.03553643375842356,0.3096819719200915,0.46799043941901075,False
SMB4_SURFACE_current_dir,test,33649,0.1557241052382081,0.26948318293683343,0.010191988622896421,0.0199907019990702,0.16591609386110454,0.2823149828594475,False
SMB4_SURFACE_current_speed,train,203991,0.2969852547569225,0.45504282745057834,0.5703318896296656,0.699039694610957,0.7217869482699888,0.8061745841939341,True
SMB4_SURFACE_current_speed,test,33651,0.1557241052382081,0.26948318293683343,0.5114956150746622,0.6242406711021118,0.6100971794264044,0.7023192360163711,True
SMB4_WATER_O2Per,train,204329,0.8425164949646949,0.9145280351814041,0.244356985762241,0.38189136628827275,0.9277983562912374,0.9452085083801354,False
SMB4_WATER_O2Per,test,33656,0.9672569526978845,0.9833559885213714,0.0,0.0,0.9672569526978845,0.9833559885213714,False
SMB4_WATER_O2ppm,train,205075,0.786607246209052,0.88055978489747,0.21686537793726127,0.35069493191071177,0.9291584674152101,0.9534386506711011,False
SMB4_WATER_O2ppm,test,33656,0.9673460898502496,0.9834020509567608,0.0,0.0,0.9673460898502496,0.9834020509567608,False
SMB4_WATER_Salinity,train,205075,0.18432994951612022,0.31126225245049016,0.019665020321128508,0.03797423805166468,0.2008966811860978,0.33016833403222906,False
SMB4_WATER_Salinity,test,33656,0.013043736629427146,0.02575157647748937,0.0,0.0,0.013043736629427146,0.02575157647748937,False
SMB4_WATER_chl-a,train,205075,0.5297637861643172,0.6925799920111293,0.9299576458688918,0.378586808266065,1.0,0.40137861039455325,False
SMB4_WATER_chl-a,test,33656,0.14409050312593033,0.25019384853967436,0.9940458469782674,0.1890499377193976,1.0,0.19007469443186964,False
SMB4_WATER_pH,train,205075,0.6929929758437553,0.818627054156885,0.25516532465307523,0.4058974219218401,0.9420250128490663,0.9690870637997885,False
SMB4_WATER_pH,test,33656,0.031194737214401923,0.06050212649197421,0.9100233430006366,0.9524690901014288,0.9211289523944259,0.958521953553421,True
SMB4_WATER_temp,train,205075,0.4057179521613919,0.5772394832655747,0.22213766656221556,0.36253165789986364,0.5192563081009296,0.6820658565268523,False
SMB4_WATER_temp,test,33656,0.013043736629427146,0.02575157647748937,0.011647254575707155,0.023026315789473683,0.024572141668647494,0.04796566424035032,False
SMB4_WATER_turbidity,train,205075,0.30607336139506913,0.4683873863763887,0.3913438833726882,0.5549997400031199,0.681416188786061,0.8015146679547669,True
SMB4_WATER_turbidity,test,33656,0.2673770888835069,0.4219377030384101,0.5542504238314362,0.7058914250462678,0.6036570598207799,0.7453648325358853,True
Sea_level_Repr_Lake,train,3671039,0.8251416322155872,0.9041693464947849,0.031031490838564792,0.059141092359321044,0.8394834890011739,0.9037114364989973,False
Sea_level_Repr_Lake,test,349913,0.4583333333333333,0.6285714285714286,1.0,0.8727272727272727,1.0,0.8727272727272727,True
Sea_level_Repr_Sea,train,3671039,0.3105693597958014,0.47393314696803085,0.007726901772846451,0.01531920753223058,0.3138511225712351,0.4773726676086469,False
Sea_level_Repr_Sea,test,349913,0.3559322033898305,0.525,0.1864406779661017,0.2953020134228188,0.4491525423728814,0.5888888888888889,False


## Pseudo-Test (SMB3/SMB4 Water Features, 2024-01-01 ~ 2024-12-31)
- Caveat: Evaluation window is within training-used period (non-independent validation).
top_folder,split,n_eval_rows,recall_1qc,f1_1qc,recall_ai_pot,f1_ai_pot,recall_1qc_plus_ai,f1_1qc_plus_ai,ai_usable,note
SMB3_WATER_O2Per,pseudo_test_2024,46497,0.9975442043222004,0.9987705925743791,0.4773084479371316,0.6460147576946088,0.9986247544204322,0.999115479115479,False,Evaluation window is within training-used period (non-independent validation).
SMB3_WATER_O2ppm,pseudo_test_2024,46780,0.8759332023575639,0.9338639576896896,0.4775049115913556,0.6462377027386333,0.9988212180746562,0.9992629354822858,False,Evaluation window is within training-used period (non-independent validation).
SMB3_WATER_Salinity,pseudo_test_2024,46780,0.9236851151598487,0.9603288062902073,0.0,0.0,0.9236851151598487,0.9603288062902073,False,Evaluation window is within training-used period (non-independent validation).
SMB3_WATER_chl-a,pseudo_test_2024,46780,0.8099508244142319,0.8948545861297539,0.8643332369106161,0.2848698636666984,1.0,0.3223760898960228,False,Evaluation window is within training-used period (non-independent validation).
SMB3_WATER_pH,pseudo_test_2024,46780,0.5432760364004044,0.7040555591954399,0.16147623862487362,0.27793247476505395,0.696056622851365,0.8205494963943024,False,Evaluation window is within training-used period (non-independent validation).
SMB3_WATER_temp,pseudo_test_2024,46780,0.9957360029662589,0.9978634463539249,0.0,0.0,0.9957360029662589,0.9977707597993686,False,Evaluation window is within training-used period (non-independent validation).
SMB4_WATER_O2Per,pseudo_test_2024,49652,0.9083098591549296,0.9519521735921471,0.0015492957746478873,0.0030937983405990716,0.9095774647887324,0.9526478831686088,False,Evaluation window is within training-used period (non-independent validation).
SMB4_WATER_O2ppm,pseudo_test_2024,50398,0.9092957746478874,0.952493360873414,0.0,0.0,0.9092957746478874,0.952493360873414,False,Evaluation window is within training-used period (non-independent validation).
SMB4_WATER_Salinity,pseudo_test_2024,50398,0.008055875233144173,0.01598299346508149,7.936822889797214e-05,0.00015872386016427918,0.008135243462042144,0.016139190678633286,False,Evaluation window is within training-used period (non-independent validation).
SMB4_WATER_chl-a,pseudo_test_2024,50398,0.05,0.09521592364101966,0.9954767726161369,0.3048385587271877,1.0,0.3060117466611799,True,Evaluation window is within training-used period (non-independent validation).
SMB4_WATER_pH,pseudo_test_2024,50398,0.04373927958833619,0.08380404642086885,0.7833404802744426,0.8777704366628627,0.8228987993138936,0.9021036549535785,True,Evaluation window is within training-used period (non-independent validation).
SMB4_WATER_temp,pseudo_test_2024,50398,0.15673275111026114,0.2709921560703198,0.30655957161981257,0.46926229508196715,0.31146809460062475,0.474991493705342,True,Evaluation window is within training-used period (non-independent validation).


## Plot Index
top_folder,split,plot_dir
SMB1_AIR_press,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB1_AIR_press\train\fig_pot
SMB1_AIR_press,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB1_AIR_press\test\fig_pot
SMB1_AIR_solar,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB1_AIR_solar\train\fig_pot
SMB1_AIR_solar,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB1_AIR_solar\test\fig_pot
SMB1_AIR_temp,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB1_AIR_temp\train\fig_pot
SMB1_AIR_temp,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB1_AIR_temp\test\fig_pot
SMB1_AIR_wind_dir,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB1_AIR_wind_dir\train\fig_pot
SMB1_AIR_wind_dir,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB1_AIR_wind_dir\test\fig_pot
SMB1_AIR_wind_max_speed,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB1_AIR_wind_max_speed\train\fig_pot
SMB1_AIR_wind_max_speed,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB1_AIR_wind_max_speed\test\fig_pot
SMB1_AIR_wind_speed,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB1_AIR_wind_speed\train\fig_pot
SMB1_AIR_wind_speed,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB1_AIR_wind_speed\test\fig_pot
SMB1_SURFACE_current_dir,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB1_SURFACE_current_dir\train\fig_pot
SMB1_SURFACE_current_dir,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB1_SURFACE_current_dir\test\fig_pot
SMB1_SURFACE_current_speed,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB1_SURFACE_current_speed\train\fig_pot
SMB1_SURFACE_current_speed,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB1_SURFACE_current_speed\test\fig_pot
SMB2_AIR_press,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB2_AIR_press\train\fig_pot
SMB2_AIR_press,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB2_AIR_press\test\fig_pot
SMB2_AIR_solar,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB2_AIR_solar\train\fig_pot
SMB2_AIR_solar,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB2_AIR_solar\test\fig_pot
SMB2_AIR_temp,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB2_AIR_temp\train\fig_pot
SMB2_AIR_temp,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB2_AIR_temp\test\fig_pot
SMB2_AIR_wind_dir,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB2_AIR_wind_dir\train\fig_pot
SMB2_AIR_wind_dir,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB2_AIR_wind_dir\test\fig_pot
SMB2_AIR_wind_max_speed,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB2_AIR_wind_max_speed\train\fig_pot
SMB2_AIR_wind_max_speed,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB2_AIR_wind_max_speed\test\fig_pot
SMB2_AIR_wind_speed,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB2_AIR_wind_speed\train\fig_pot
SMB2_AIR_wind_speed,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB2_AIR_wind_speed\test\fig_pot
SMB2_SURFACE_current_dir,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB2_SURFACE_current_dir\train\fig_pot
SMB2_SURFACE_current_dir,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB2_SURFACE_current_dir\test\fig_pot
SMB2_SURFACE_current_speed,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB2_SURFACE_current_speed\train\fig_pot
SMB2_SURFACE_current_speed,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB2_SURFACE_current_speed\test\fig_pot
SMB3_SURFACE_current_dir,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_SURFACE_current_dir\train\fig_pot
SMB3_SURFACE_current_dir,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_SURFACE_current_dir\test\fig_pot
SMB3_SURFACE_current_speed,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_SURFACE_current_speed\train\fig_pot
SMB3_SURFACE_current_speed,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_SURFACE_current_speed\test\fig_pot
SMB3_WATER_O2Per,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_WATER_O2Per\train\fig_pot
SMB3_WATER_O2Per,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_WATER_O2Per\test\fig_pot
SMB3_WATER_O2ppm,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_WATER_O2ppm\train\fig_pot
SMB3_WATER_O2ppm,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_WATER_O2ppm\test\fig_pot
SMB3_WATER_Salinity,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_WATER_Salinity\train\fig_pot
SMB3_WATER_Salinity,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_WATER_Salinity\test\fig_pot
SMB3_WATER_chl-a,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_WATER_chl-a\train\fig_pot
SMB3_WATER_chl-a,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_WATER_chl-a\test\fig_pot
SMB3_WATER_pH,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_WATER_pH\train\fig_pot
SMB3_WATER_pH,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_WATER_pH\test\fig_pot
SMB3_WATER_temp,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_WATER_temp\train\fig_pot
SMB3_WATER_temp,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_WATER_temp\test\fig_pot
SMB3_WATER_turbidity,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_WATER_turbidity\train\fig_pot
SMB3_WATER_turbidity,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB3_WATER_turbidity\test\fig_pot
SMB4_SURFACE_current_dir,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_SURFACE_current_dir\train\fig_pot
SMB4_SURFACE_current_dir,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_SURFACE_current_dir\test\fig_pot
SMB4_SURFACE_current_speed,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_SURFACE_current_speed\train\fig_pot
SMB4_SURFACE_current_speed,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_SURFACE_current_speed\test\fig_pot
SMB4_WATER_O2Per,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_WATER_O2Per\train\fig_pot
SMB4_WATER_O2Per,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_WATER_O2Per\test\fig_pot
SMB4_WATER_O2ppm,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_WATER_O2ppm\train\fig_pot
SMB4_WATER_O2ppm,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_WATER_O2ppm\test\fig_pot
SMB4_WATER_Salinity,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_WATER_Salinity\train\fig_pot
SMB4_WATER_Salinity,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_WATER_Salinity\test\fig_pot
SMB4_WATER_chl-a,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_WATER_chl-a\train\fig_pot
SMB4_WATER_chl-a,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_WATER_chl-a\test\fig_pot
SMB4_WATER_pH,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_WATER_pH\train\fig_pot
SMB4_WATER_pH,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_WATER_pH\test\fig_pot
SMB4_WATER_temp,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_WATER_temp\train\fig_pot
SMB4_WATER_temp,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_WATER_temp\test\fig_pot
SMB4_WATER_turbidity,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_WATER_turbidity\train\fig_pot
SMB4_WATER_turbidity,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\SMB4_WATER_turbidity\test\fig_pot
Sea_level_Repr_Lake,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\Sea_level_Repr_Lake\train\fig_pot
Sea_level_Repr_Lake,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\Sea_level_Repr_Lake\test\fig_pot
Sea_level_Repr_Sea,train,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\Sea_level_Repr_Sea\train\fig_pot
Sea_level_Repr_Sea,test,H:\codex siwha AI anomaly detection\output\bilstm_codex\plots\Sea_level_Repr_Sea\test\fig_pot


## Manifest Reference
- Rows: 36
- File: `output/bilstm_codex/manifests/experiment_manifest.csv`
