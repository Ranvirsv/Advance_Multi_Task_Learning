# MTL Model Predictions Report

## 1. Validation Set (With True Labels)
> **Note:** The validation set contains true ground-truth targets for Toxicity and Engagement.

### Sample ID: `276010`
**Comment:**
> “So, do you think it would be acceptable for a woman to proposition someone for sex, strip down and lather themselves in soap?” 
> That sounds disgusting! You're sick!

- **True Toxicity:** Toxic
- **True Engagement:** Ignored

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | **0.9029** | 0.3503 |
| BasicMoE | **0.5110** | **0.5052** |
| BasicMMoE | **0.5373** | 0.4712 |

---

### Sample ID: `343297`
**Comment:**
> "I shot a man in Reno, just to watch him die." Jackass.

- **True Toxicity:** Toxic
- **True Engagement:** Clicked

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | **0.5810** | 0.4149 |
| BasicMoE | **0.5045** | **0.5109** |
| BasicMMoE | **0.5210** | 0.4751 |

---

### Sample ID: `319574`
**Comment:**
> It is important to remind people why such shameful behavior such as occurred in Toksook Bay is a violation of federal law.
> 
> Numerous whale species are endangered because of the behavior of humans.  Treaties have been signed by the US and other nations to prevent the EXTINCTION of whales.  When outlaws decided to go kill whales other nations can justify this bad behavior and declare that they too, have a right to continue killing whales- until they go extinct.
> 
> The Northern Right Whale, the Southern Right Whale, the Bowhead Whale, the Fin Whale, the Sei Whale, the Vaquita Whale, the Baiji Whale, the Sperm Whale, and the Humpback Whale are ALL ENDANGERED.
> 
> Some of these species only have a few hundred living members left.
> 
> Those who killed this endangered whale must be prosecuted.

- **True Toxicity:** Safe
- **True Engagement:** Clicked

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | **0.6839** | 0.4740 |
| BasicMoE | 0.4965 | **0.5052** |
| BasicMMoE | **0.5162** | 0.4771 |

---

### Sample ID: `275857`
**Comment:**
> NE Portlander wasn't talking about the obsession with "Authentic Portland" Or stupid Old vs New Portland.  But the rapidly rising obesity rates in the city.

- **True Toxicity:** Toxic
- **True Engagement:** Ignored

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | **0.7387** | 0.2228 |
| BasicMoE | **0.5086** | **0.5003** |
| BasicMMoE | **0.5279** | 0.4692 |

---

### Sample ID: `282022`
**Comment:**
> Yeah, I see Russia, like a drunk and dangerous snow machiner buzzing Iditarod mushers. Very tRUMP & cRUSE like of you, Mr. Putin.

- **True Toxicity:** Toxic
- **True Engagement:** Ignored

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | **0.7940** | 0.4905 |
| BasicMoE | **0.5049** | **0.5085** |
| BasicMMoE | **0.5247** | 0.4775 |

---

### Sample ID: `333450`
**Comment:**
> Mr. Severson,
> 
> Thank you for your service.  Both as a soldier, and as a warrior for Veterans in Alaska.
> 
> And, thank you for writing this piece.

- **True Toxicity:** Safe
- **True Engagement:** Ignored

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.0366 | 0.4605 |
| BasicMoE | 0.4668 | **0.5108** |
| BasicMMoE | 0.4820 | 0.4778 |

---

### Sample ID: `246099`
**Comment:**
> Drinking lots of beer?

- **True Toxicity:** Safe
- **True Engagement:** Ignored

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.2706 | 0.3007 |
| BasicMoE | 0.4801 | **0.5044** |
| BasicMMoE | 0.4926 | 0.4659 |

---

### Sample ID: `287123`
**Comment:**
> Majority leader John Coghill, R-North Pole. “Everyone wants their stuff to happen, right?”
> 
> How about the real issues, Coghill? Idiots.

- **True Toxicity:** Toxic
- **True Engagement:** Ignored

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.4904 | 0.4178 |
| BasicMoE | 0.4967 | **0.5043** |
| BasicMMoE | **0.5203** | 0.4764 |

---

### Sample ID: `308550`
**Comment:**
> I'm glad we are seeing some blowback on these events. Some of these "charity" events are complete nonsense.

- **True Toxicity:** Toxic
- **True Engagement:** Clicked

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.2812 | 0.4783 |
| BasicMoE | 0.4832 | **0.5085** |
| BasicMMoE | **0.5065** | 0.4783 |

---

### Sample ID: `302435`
**Comment:**
> TRUMP  2K16!!!

- **True Toxicity:** Safe
- **True Engagement:** Clicked

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | **0.7191** | 0.3789 |
| BasicMoE | **0.5003** | **0.5038** |
| BasicMMoE | **0.5166** | 0.4732 |

---

## 2. True Test Set (Inference Only)
> **Note:** The Kaggle test set does *not* contain true labels. These metrics represent raw inference probabilities.

### Sample ID: `7101679`
**Comment:**
> "I just want to get rail done no matter what the consequences are." Caldwell is dangerous and wants us to pay for his ineptness and ego. Tear it down.

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.3344 | **0.6121** |
| BasicMoE | 0.4863 | **0.5153** |
| BasicMMoE | **0.5111** | 0.4847 |

---

### Sample ID: `7102988`
**Comment:**
> Just because armed people (victims, witnesses, and bystanders alike) couldn't have prevented or mitigated this particular shooting doesn't mean the notion is invalid in countless other scenarios.

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.3067 | 0.4541 |
| BasicMoE | 0.4920 | **0.5066** |
| BasicMMoE | 0.4982 | 0.4783 |

---

### Sample ID: `7102773`
**Comment:**
> You are looking at the past with rose-colored glasses. When the Irish Poles and Jews first arrived, they were not considered to be "from the highly educated classes of Europe, noted scholars and business people." The reaction to them was just as full of fear-mongering and ignorance as the reaction to today's immigrants.

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | **0.6100** | 0.2914 |
| BasicMoE | **0.5002** | **0.5047** |
| BasicMMoE | **0.5218** | 0.4733 |

---

### Sample ID: `7104764`
**Comment:**
> Thanks for the wit!  Snorted out loud :)

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | **0.5479** | 0.2252 |
| BasicMoE | **0.5076** | 0.4979 |
| BasicMMoE | **0.5171** | 0.4638 |

---

### Sample ID: `7104131`
**Comment:**
> THE DEMOCRAT PARTY IS THE PARTY OF GREED FROM TOP TO BOTTOM . 
> IT IS UNFORTUNATE , BUT GREED SELLS THESE DAYS . 
> ~
> IF WE ALL VOTED FOR THE GOOD OF THE COUNTRY INSTEAD OF THE GOOD OF OURSELVES , 
> THE COUNTRY WOULD BE PROSPEROUS FOR EVERYONE .

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.3642 | 0.3917 |
| BasicMoE | 0.4888 | **0.5066** |
| BasicMMoE | **0.5177** | 0.4773 |

---

### Sample ID: `7103625`
**Comment:**
> Wow...That's a beautiful display of human awareness. Funny...Drumpf didn't feel he was humiliating anyone either. I see a pattern developing.
> 
> I await your dismissive, divisive, fact free, ad homonym infused faux anger response as always.

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | **0.7833** | 0.4252 |
| BasicMoE | 0.4989 | **0.5055** |
| BasicMMoE | **0.5226** | 0.4741 |

---

### Sample ID: `7104924`
**Comment:**
> Why don't they all just quit and go to work at Hooters?

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | **0.7937** | 0.3073 |
| BasicMoE | **0.5129** | **0.5095** |
| BasicMMoE | **0.5299** | 0.4733 |

---

### Sample ID: `7103847`
**Comment:**
> Coaches who engage in emotional rants along the sidelines embarrass themselves, their players and their schools.  There is absolutely nothing to be gained by this behavior.  It teaches poor sportsmanship and encourages players to ultimately to the same.  Coach needs to learn how to dial it down.

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.3181 | 0.3034 |
| BasicMoE | 0.4835 | **0.5007** |
| BasicMMoE | **0.5046** | 0.4705 |

---

### Sample ID: `7103126`
**Comment:**
> It's a hit job to be sure, but this isn't the 50s or 60s where we only have one source of information available.
> 
> People today can see the  media bias for what it is, as evidenced by this last election.
> 
> Expect more people to wake up in the coming years. Their stranglehold on information is over.

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.0749 | 0.4561 |
| BasicMoE | 0.4713 | **0.5101** |
| BasicMMoE | 0.4982 | 0.4806 |

---

### Sample ID: `7104728`
**Comment:**
> As expected.  And most of them are back before the day is out.

| Model | Toxicity Prob | Engagement Prob |
|-------|---------------|-----------------|
| SharedBottomModel | 0.1662 | 0.3574 |
| BasicMoE | 0.4822 | **0.5064** |
| BasicMMoE | 0.4983 | 0.4711 |

---

