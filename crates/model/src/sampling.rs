use tritium_framework::Random;

const LOGIT_ONE: f32 = 256.0;

pub fn sample(logits: &[i32], temperature: f32, random: &mut Random) -> usize {
    let best = logits
        .iter()
        .enumerate()
        .max_by_key(|&(index, &logit)| (logit, std::cmp::Reverse(index)))
        .map_or(0, |(index, _)| index);
    if temperature <= 0.0 {
        return best;
    }
    let peak = logits[best] as f32;
    let weights: Vec<f32> = logits
        .iter()
        .map(|&logit| ((logit as f32 - peak) / LOGIT_ONE / temperature).exp())
        .collect();
    let mut threshold = random.uniform() * weights.iter().sum::<f32>();
    for (index, weight) in weights.iter().enumerate() {
        if threshold < *weight {
            return index;
        }
        threshold -= weight;
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_temperature_picks_first_maximum() {
        assert_eq!(sample(&[3, 9, 9, -4], 0.0, &mut Random::new(1)), 1);
    }

    #[test]
    fn sampling_follows_the_logits() {
        let mut random = Random::new(2);
        let logits = [0, 256 * 3, 0, -256 * 50];
        let picks: Vec<usize> = (0..2000).map(|_| sample(&logits, 1.0, &mut random)).collect();
        let favored = picks.iter().filter(|&&pick| pick == 1).count();
        assert!((1700..1950).contains(&favored), "{favored}");
        assert!(!picks.contains(&3));
    }
}
