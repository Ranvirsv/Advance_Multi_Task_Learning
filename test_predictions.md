# MTL Model Predictions Report

## 1. Validation Set (With True Labels)
> **Note:** The validation set contains true ground-truth targets for Toxicity and Engagement.

### Sample ID: `329724`
**Comment:**
> Obama has a double standard for everything. He has a big hypocrite. Only 6 more months and I want have to look at his ugly mug.

- **True Toxicity:** Toxic
- **True Engagement:** Ignored

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | **0.8479** | 0.3724 |
| BasicMoE | **0.5148** | **0.5061** |
| BasicMMoE | **0.5012** | 0.4895 |

---

### Sample ID: `336583`
**Comment:**
> This show stinks to high heaven and the Browns stink also

- **True Toxicity:** Toxic
- **True Engagement:** Ignored

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | **0.6507** | 0.3371 |
| BasicMoE | **0.5120** | **0.5014** |
| BasicMMoE | 0.4969 | 0.4898 |

---

### Sample ID: `363071`
**Comment:**
> Leave me out of your "we".  I have not voted for that clown for a lot of years.

- **True Toxicity:** Toxic
- **True Engagement:** Clicked

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | **0.8963** | 0.3406 |
| BasicMoE | **0.5293** | **0.5063** |
| BasicMMoE | **0.5065** | 0.4890 |

---

### Sample ID: `283812`
**Comment:**
> Alaska's national embarrassment spouts nonsense again. Thanks Bible Barbie.

- **True Toxicity:** Toxic
- **True Engagement:** Clicked

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | **0.8314** | **0.5728** |
| BasicMoE | **0.5203** | **0.5168** |
| BasicMMoE | 0.4960 | 0.4953 |

---

### Sample ID: `312383`
**Comment:**
> This has been an ongoing issue and will not be solved until you have the systems in place to identify the individual needs of those on the street.  Many are self induced as we all know and many don't want you meddling in their business. I have a hard time believing any of the sign holders would willingly get into a muni van to go work.  It will be more like "take a hike buddy I don't want your work I make plenty right here"!

- **True Toxicity:** Safe
- **True Engagement:** Ignored

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.1407 | 0.4510 |
| BasicMoE | 0.4876 | **0.5095** |
| BasicMMoE | 0.4779 | 0.4931 |

---

### Sample ID: `282237`
**Comment:**
> Why, with a theatrical version in a park of course. :) https://www.youtube.com/watch?v=trHzYyuRrCw

- **True Toxicity:** Safe
- **True Engagement:** Ignored

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.2710 | 0.3369 |
| BasicMoE | 0.4922 | **0.5074** |
| BasicMMoE | 0.4823 | 0.4920 |

---

### Sample ID: `266259`
**Comment:**
> I posted national data in response to your assertions about national polls...that data contradicted your assertions...so now you switch to Oregon...and add in personal insults.
> Your ignorance regarding civil discussion is profound, and your arrogance is over the top.

- **True Toxicity:** Toxic
- **True Engagement:** Ignored

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | **0.7261** | 0.1912 |
| BasicMoE | **0.5178** | 0.4961 |
| BasicMMoE | 0.4995 | 0.4800 |

---

### Sample ID: `263636`
**Comment:**
> "...and Gollum kneeled at Frodo's feet, pawing at his knees, hissing 'Nice massssster..."

- **True Toxicity:** Safe
- **True Engagement:** Ignored

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.1363 | 0.3988 |
| BasicMoE | 0.4935 | **0.5071** |
| BasicMMoE | 0.4754 | 0.4917 |

---

### Sample ID: `277271`
**Comment:**
> wondering if Rebecka Logan and Kara Moriarty were on the second floor with their i phones,... connected to Chenaults?

- **True Toxicity:** Safe
- **True Engagement:** Clicked

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.0266 | 0.2694 |
| BasicMoE | 0.4662 | 0.4950 |
| BasicMMoE | 0.4596 | 0.4861 |

---

### Sample ID: `253697`
**Comment:**
> Their right is to break into other peoples homes and steal those things. No sense burdening the taxpayer anymore than free room and board. as for their drugs, that's what professionally begging on street corners is for. They need a job.

- **True Toxicity:** Safe
- **True Engagement:** Ignored

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | **0.6168** | **0.5412** |
| BasicMoE | **0.5078** | **0.5108** |
| BasicMMoE | 0.4954 | 0.4961 |

---

## 2. True Test Set (Inference Only)
> **Note:** The Kaggle test set does *not* contain true labels. These metrics represent raw inference probabilities.

### Sample ID: `7103696`
**Comment:**
> A letter of reprimand would have been more appropriate.

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.3896 | 0.4050 |
| BasicMoE | 0.4951 | **0.5078** |
| BasicMMoE | 0.4842 | 0.4910 |

---

### Sample ID: `7104876`
**Comment:**
> Not taking anything from clarke but enough about clarke - great player but also overrated by leaf fans

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.0706 | 0.2501 |
| BasicMoE | 0.4752 | 0.4987 |
| BasicMMoE | 0.4701 | 0.4853 |

---

### Sample ID: `7105482`
**Comment:**
> My take away from the author seems different than most here but here we go ....
> Peter Stockland writes to further his religious beliefs and equates "freedom" with the pre-eminence of religion over state.  He quotes a theology friend "the freedoms of religion and conscience are and must be the first freedoms to which all others are bound."  Whoa there mister!  Conscience yes, but after that we are supposed to have an overriding principle of the separation of church and state in our modern country and for darn good reasons too.  Organized religions, as presided over by their 'leaders' and allies have too often been the creators and users of oppression, subjugation, and many of the wars over the past two thousand years. Today they continue to provide the tools of propaganda for whomever wields power, be that a vice president, an ayatollah, or the power blocs behind them.  I can't speak for their reasons but the citizens VOTED.  It seems Peter prefers religious groups have the final say.

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.1405 | 0.4662 |
| BasicMoE | 0.4871 | **0.5117** |
| BasicMMoE | 0.4778 | 0.4927 |

---

### Sample ID: `7104257`
**Comment:**
> What did P T Barnum say?

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.1729 | 0.2699 |
| BasicMoE | 0.4881 | **0.5039** |
| BasicMMoE | 0.4800 | 0.4892 |

---

### Sample ID: `7103863`
**Comment:**
> I would have laughed if I'd finished crying over the other articles today about Trudeau's looming tax bombs.

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.4821 | 0.3917 |
| BasicMoE | **0.5010** | **0.5047** |
| BasicMMoE | 0.4913 | 0.4926 |

---

### Sample ID: `7106530`
**Comment:**
> If you don’t think Vegas will come back to earth I have a bridge in Brooklyn to sell you.
> 
> Your “yesterday’s stuff” is just inane and devoid of any basis in reality in regard to the current edition of the Leafs. And the one UFA vet they signed is doing just fine and seems badly missed by his struggling former team.

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.1488 | 0.2980 |
| BasicMoE | 0.4932 | **0.5007** |
| BasicMMoE | 0.4755 | 0.4880 |

---

### Sample ID: `7103939`
**Comment:**
> Or, in the case of Clinton, a centrist.

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.2904 | 0.4262 |
| BasicMoE | 0.4983 | **0.5080** |
| BasicMMoE | 0.4795 | 0.4896 |

---

### Sample ID: `7099785`
**Comment:**
> Doesn't change the fact that he was dealt the worst hand of any incoming President ever. Besides the blatant racism he has faced and the disastrous economy he inherited, he has led the country with class and intelligence. Is he perfect? Of course not. But this Republican rhetoric that continually claims he has been a "disaster" is just "trumped" up hot air. Are we better off than we were 8 years ago? Without question.

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.4700 | 0.4101 |
| BasicMoE | **0.5111** | **0.5073** |
| BasicMMoE | 0.4877 | 0.4920 |

---

### Sample ID: `7097637`
**Comment:**
> 'Have nothing'? We have both houses, the Presidency and the courts.  I would call that a 'full house'!

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.0751 | 0.3674 |
| BasicMoE | 0.4848 | **0.5113** |
| BasicMMoE | 0.4717 | 0.4925 |

---

### Sample ID: `7106695`
**Comment:**
> Eric
> Cut the guy a little slack, he's so giddy now thinking he's won, it'll take 6 months or a year for the reality of the what's happening to the country to sink in.
> charlie

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.4631 | **0.5046** |
| BasicMoE | **0.5070** | **0.5148** |
| BasicMMoE | 0.4880 | 0.4969 |

---

