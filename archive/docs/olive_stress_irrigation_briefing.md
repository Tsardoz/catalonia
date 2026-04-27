# Olive Climate Stress and Irrigation Timing: A Briefing for Supervisors

**Project:** PhD Research — AI in Horticultural Irrigation  
**Document purpose:** Explain what the Catalan olive analysis found and how it connects to practical irrigation scheduling  
**Audience:** This document is intended for supervisors with different backgrounds. It does not assume specialist knowledge of machine learning or agronomy. It does not make irrigation recommendations — the data are not sufficient for that. It explains what was done, what was found, and how the findings relate to known olive agronomy and to the longer-term thesis objectives.  
**Date:** April 2026

---

## 1. What the Catalan Analysis Did

This work assembled a dataset linking rainfed olive yields across 36 Catalan administrative regions (comarques) to daily weather conditions from 2016 to 2024. The Catalan data were chosen because they represent one of the very few publicly available sources that report crop yield at a regional administrative level with sufficient temporal coverage to support climate analysis. Most yield datasets are either privately held, aggregated at national scale, or not linked to spatial units that can be matched to gridded weather products.

Having identified the Catalan yield records, a separate weather dataset had to be sourced and spatially aligned to the same regional boundaries. AgERA5 — a gridded daily agrometeorological reanalysis product from the Copernicus Climate Data Store — was downloaded and each grid cell assigned to the comarca it falls within. Aligning two independently sourced datasets across matching spatial units required substantial construction work, and is one reason this kind of regional perennial crop analysis is uncommon in the literature.

Yield records from Generalitat de Catalunya administrative data were matched to gridded AgERA5 climate, with weather exposure weighted by actual olive farm locations and corrected for elevation. The pipeline produced 320 comarca-year observations.

A key design choice was the selection of olive as the target crop. The administrative yield data cover many woody crop types, but farm-level irrigation status is not recorded — it is not possible to determine from the available data whether an individual farm is irrigated or rainfed. For most crop types this creates a significant confound: irrigated farms buffer atmospheric stress, so yield variation reflects management decisions as much as weather. Olives are chosen because rainfed cultivation dominates the crop type in Catalonia; the proportion of irrigated olive production is small enough that comarca-level yield is predominantly driven by rainfall and atmospheric conditions rather than irrigation management. Rainfed dominance does not eliminate irrigation confounding entirely, but it minimises it sufficiently to recover a weather-driven signal. Private farm-level datasets recording irrigation status and volume would allow this confound to be removed more cleanly and would be expected to produce stronger and more precise climate-yield associations.

Two analyses were applied. First, phenological-window regression identified which seasonal climate summaries predicted yield across the three main olive growth stages. Second, rolling-window timing scans fitted a simple model for each 14-day window across a candidate period, to find when during the season weather has the largest effect on yield. Summer windows (June to August) were used for the Tmax and VPD scans, where acute heat and dry-air stress are expected. The crude water balance scan (P − ET₀) was applied across the full agronomic year (December to November) to capture both winter recharge and summer deficit signals. Drainage was omitted from the water balance as it is unknown.

The primary finding is that a **late July to late August window** carries the greatest yield cost per day of heat or dry-air stress. The clearest indicator was the count of days with maximum temperature at or above 32°C. Vapour pressure deficit (VPD) above 3.0 kPa produced a consistent result over the same period, supporting the temperature finding rather than replacing it. Stress during early July, by contrast, shows a mild positive or neutral association with yield — consistent with the known olive agronomy practice of allowing mild water deficit during pit hardening.

These are exploratory associations from observational data, not confirmed causal mechanisms. They identify *when* yield is most sensitive to weather, which is the question irrigation scheduling needs to answer.

---

## 2. Why Stress Causes Yield Loss

The pathway from weather to yield operates through plant physiology. When the air is hot and dry, a large vapour pressure gradient exists between the humid interior of the leaf and the surrounding air. The tree responds by closing its stomata to limit water loss. This is the correct short-term survival response, but it has a cost: stomata are also the entry point for CO₂. Stomatal closure therefore simultaneously stops photosynthesis.

Importantly, the severity of this response depends on soil moisture availability as well as atmospheric conditions. When the root zone is adequately supplied with water, the tree can meet atmospheric demand through transpiration without closing its stomata, even under high temperatures or dry air. Stress occurs when atmospheric demand and insufficient soil water supply combine. Precipitation and water balance variables therefore appear alongside temperature in the analysis, and irrigation — by maintaining soil moisture — can buffer the effect of hot, dry weather on yield.

When the tree suppresses photosynthesis, it limits the carbohydrates available for fruit development. The timing of this suppression within the growth cycle determines how much yield is lost. During active fruit and oil accumulation — from late July onward in Mediterranean olives — carbohydrate demand from the developing fruit is high. Stress at this stage directly limits the resource the tree needs to fill its yield. During the earlier pit hardening stage, the fruit is not yet accumulating oil and is less metabolically demanding, so stress is more tolerable.

Physiological sensitivity explains why the empirical signal from the rainfed analysis points to the late July to late August window — it is not an artefact of the data but a reflection of a known biological sensitivity period.

---

## 3. Why Weather Is the Right Input for Scheduling

Crop water status can be monitored through a range of indicators. Direct plant measurements such as stem water potential, stomatal conductance, and trunk diameter fluctuation provide a close view of stress at the individual tree level. Soil moisture — measured as volumetric water content or tension — reflects the water supply available to roots and is widely used in irrigation scheduling. Each approach is valuable where sensors are available and maintained.

Weather-based monitoring offers a complementary and practically important perspective. Temperature, precipitation, and humidity are the primary observed variables. Derived quantities such as VPD and reference evapotranspiration (ET₀) are calculated from these inputs and offer two practical advantages: they are more directly interpretable in terms of the biological stress mechanism, and they reduce the number of predictors needed in a model by combining correlated raw variables into a single physically meaningful quantity. A machine learning model could in principle work directly from raw weather inputs without calculating VPD or ET₀, but including these derived variables improves interpretability and makes the model's behaviour easier to explain and validate. That said, VPD alone does not determine stress — its effect on stomata is moderated by soil moisture, which is why precipitation and water balance variables remain important alongside it.

Gridded weather data are freely available worldwide and require no on-farm instrumentation, making weather-based scheduling accessible to growers who do not have soil or plant sensors installed. Where sensors are available, the approaches work well together: weather data indicate when a high-consequence period is approaching, while plant and soil measurements confirm whether the response was sufficient.

---

## 4. Practical Interventions and When to Use Them

The following discusses types of intervention that the timing findings are consistent with, based on established olive agronomy. This is illustrative — the data are not sufficient to recommend specific practices. The late July to late August window is where the data show the strongest association between heat stress and yield loss, and where established agronomy also identifies the highest sensitivity. The early July period, where mild stress is tolerable, is recognised in regulated deficit irrigation (RDI) practice as an opportunity to conserve water before the higher-sensitivity window opens.

### 4.1 Shade

Shade cloth reduces canopy temperature directly, lowering both the heat load on the developing fruit and the saturation vapour pressure that drives VPD. A cooler canopy faces a smaller leaf-to-air vapour gradient, so stomata remain more open and photosynthesis continues. Shade is most efficiently deployed in the late July to late August window, where the data show the highest per-day yield cost of heat exposure.

### 4.2 Humidity Management

When the air is dry as well as hot, the atmospheric demand on the tree is higher. Increasing ambient humidity around the canopy reduces this demand independently of temperature. In greenhouse production this can be managed through fogging or misting systems where the enclosed environment makes such approaches practical and efficient. In open orchards, under-canopy microsprinklers serve a similar purpose at the root zone level — cooling the soil surface and reducing the atmospheric demand experienced by the lower canopy — without the water losses associated with overhead application in open-air conditions.

### 4.3 Irrigation During Heat Events

Irrigation during extreme heat episodes serves two functions. It replenishes soil water so roots can meet transpiration demand without stomatal closure, and evaporative cooling from wet soil and canopy surfaces can reduce ambient temperature. These effects are most relevant during the late July to August window where the data show the highest yield sensitivity, and are consistent with why irrigation scheduling practice generally protects this period rather than treating it as an opportunity for deficit management.

### 4.4 Winter Soil Recharge

The full-year water balance analysis (precipitation minus ET₀) shows a positive association between yield and wet conditions in late December to early January. For rainfed systems this is simply winter rainfall. For irrigated orchards it points to soil profile recharge as a meaningful early-season lever. Filling the soil water store before the season begins extends the buffer available during the summer stress period and reduces the volume of reactive in-season irrigation needed when the high-consequence window arrives. Winter irrigation is therefore not a substitute for summer management — it reduces the pressure on it.

---

## 5. Alignment with Known Best Practice

The data are not sufficient to make irrigation recommendations. The linear regression models used here are simple screening tools — their purpose is to establish whether weather variables are significant predictors of yield at all, and whether the timing of strongest association is consistent with what olive agronomy already knows. On both counts the answer is yes.

The late July to early August sensitive window identified by the Tmax and VPD scans aligns with the known high-consequence period in olive phenology. The neutral or mildly positive early July signal is consistent with established regulated deficit irrigation (RDI) practice, which deliberately allows mild water deficit during pit hardening without yield penalty. The positive winter water balance signal is consistent with the known importance of soil profile recharge before the season. The data-driven associations are not generating new agronomic knowledge — they are confirming that weather-based signals correlate with patterns that practitioners and researchers have identified through other means. That is the appropriate claim for this stage of analysis.

**Limitations of the current approach and planned improvements:**

The current models use single predictors with comarca fixed effects. Several known sources of imprecision remain. Phenological stage boundaries are defined by fixed calendar dates, but in practice stage timing varies with temperature accumulation across years and across comarques at different elevations. Growing degree days (GDD) would allow stages to be located more precisely in calendar time, which is likely to sharpen the timing signal. Irrigation confounding, while minimised by the choice of olive, is not fully removed; additional approaches to isolate the rainfed signal would improve confidence. Including multiple predictors and their interactions — for example, temperature combined with precipitation, which together better represent water stress than either alone — is also expected to improve model performance.

In the longer term, the intent is to apply a Temporal Fusion Transformer (TFT) trained on CropClimateX annual crop data and fine-tuned on this Catalan dataset using low-rank adaptation (LoRA). This would move from the simple linear associations demonstrated here toward a model capable of capturing nonlinear interactions, phenological dynamics, and cross-crop transfer — the foundation for the irrigation scheduling application the thesis targets. The linear regressions are a necessary first step: they establish that meaningful weather-yield signal exists in this data before more complex modelling is justified.

---

## 6. The Connection to the Broader Thesis

The Catalan analysis is not the irrigation decision system itself. It is the empirical foundation that justifies the approach the thesis proposes to transfer. By demonstrating that weather-based stress signals produce measurable yield consequences at specific growth stages in a rainfed Mediterranean perennial crop, it establishes that the causal pathway — hot and dry conditions combined with insufficient soil moisture cause stomatal closure, stomatal closure suppresses photosynthesis, photosynthesis suppression reduces yield — is legible in historical observational data across different crop types and geographies.

Training a predictive model on large-scale rainfed crop data is therefore justified, with the expectation that the timing and directional signal will transfer to an irrigated orchard context. The irrigation scheduling application is the use case that motivated the whole transfer learning design: if the model correctly identifies when atmospheric stress is most damaging to yield, it can identify when irrigation is most valuable.

---

## 7. Future Analytical Directions

The full-season water balance scan (cumulative P − ET₀ across the agronomic year) was the only variable examined at whole-season scale. The same scan applied to VPD, Tmax, and cumulative ET₀ independently would establish whether the late July to August signal identified in the rolling-window analysis also emerges when the full agronomic year is examined without phenological windowing. This would serve as a consistency check and may reveal additional signals outside the June to August candidate window currently examined.

Tmin has not been analysed. The current timing scans focus on Tmax as the heat stress indicator, but warm nights are an independent physiological stressor — high overnight temperatures increase respiration costs without enabling photosynthesis, reducing net carbon accumulation. Night temperature thresholds relevant to Mediterranean fruit crops are documented in the literature and represent a straightforward extension of the existing rolling-window framework. A Tmin timing scan using the same 14-day rolling approach would establish whether warm nights in the sensitive window carry additional explanatory power beyond Tmax alone, or whether their co-occurrence with hot days means they add little independent information.

Both extensions are straightforward to implement given the existing pipeline infrastructure, and each would establish whether the identified sensitive window holds across a broader set of atmospheric stress indicators.

---

*Document prepared as part of PhD candidature in AI in Horticultural Irrigation, Deakin University.*

*This document was produced by Claude (Anthropic) under the direction of the candidate. The analysis it describes was conducted by the candidate using Python code developed with Claude assistance. The document was drafted iteratively through a directed conversation in which the candidate provided corrections, context, and framing decisions throughout.*
