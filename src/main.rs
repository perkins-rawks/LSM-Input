/*
   If you do NOT want transposed data in the file this program writes, use:
   > cargo run notranspose

   Making ambient noise (abstractly) for our summer LSM project.

   A square grid of black and white pixels represents one frame (as in, for
   example, one millisecond) of sound in an environment.

   The first frame will be empty, and thereafter, each pixel will have a
   small probability of toggling to white from black or vice versa. This
   abstractly represents a low level of sound in the surroundings. The LSM
   is thought to be sitting in this environment, processing each sound, but not
   necessarily reacting at every moment.

   A black pixel will be stored as a 1 in the frame.

   To do list:
    o Insert bursts of meaningful Symbols into the sound frame flow.
    o Stop using strings and use bytes.
*/

#![allow(dead_code)]

use rand::prelude::*; // For use with random numbers
use std::collections::HashMap; // Allows the HashMap data structure
use std::env;
// use std::fs::remove_file;
use std::fs::File; // For file input
use std::io::prelude::*; // For writing to a file
use std::io::BufReader; // For reading standard input line by line // Allows us to use command line arguments (e.g. cargo run transpose)

/* For creating the 'soundscape' vector */
use rand_distr::{Distribution, Normal, Poisson};
const POISSON_MEAN: f32 = 0.001; // Average milliseconds between symbol bursts
const BURST_MEAN: f32 = 30.; // Average length of a symbol burst
const BURST_SD: f32 = 0.001; // Standard deviation for length of a symbol burst

/* For overall control and specs for the ambient noise */
const SOUND_SIZE: usize = 9; // Each frame of sound will be square with this side length
const NUM_FRAMES: isize = 30; // Number of sound frames that we want to generate
const NUM_TRAIN_SETS: usize = 1; // Number of random training sets to generate
const PROB_TO_BLACK: f64 = 0.05; // Probability that a white pixel turns black
const PROB_TO_WHITE: f64 = 0.50; // Probability that a black pixel turns white

/* For applying noise to the final spike train. */
const PROB_NOISE: f64 = 0.0; // Probability that each pixel in the final sound frame will switch 0 <-> 1

fn main() -> std::io::Result<()> {
    // If the command line call had an argument (as in: cargo run xxx), grab xxx:
    let args: Vec<String> = env::args().collect();
    let mut transpose = true;
    if args.len() > 1 {
        if args[1] == "notranspose" {
            transpose = false;
        }
    }

    // Creates training input and labels
    let mut file = File::create("sample_input.txt")?;
    let mut file_actions = File::create("sample_input_actions.txt")?;

    // let symbols = load_symbols();
    let num_meanings = 3;
    let mut symbol_id = 0;
    let file_name = "./src/symbols_9x9.txt";
    let symbols = load_symbols(file_name);
    for i in 0..NUM_TRAIN_SETS {
        create_input_file(
            transpose,
            &mut file,
            &mut file_actions,
            (i % (num_meanings + 1)) as u8,
            &symbols,
        )?;
        writeln!(file, "#")?;
        // writeln!(file_actions, "#")?;
        symbol_id = (symbol_id + 1) % num_meanings;
    }

    Ok(())
}

fn load_symbols(file_name: &str) -> HashMap<String, Symbol> {
    // Grab the contents of the input file:
    let f = File::open(file_name).expect("This file does not exist!");
    let file = BufReader::new(f);

    let mut lines_iter = file.lines().map(|l| l.unwrap());

    // The first line of the input file is the desired SOUND_SIZE:
    let ss = lines_iter.next();

    match ss {
        Some(i) => {
            let size = i
                .parse::<usize>()
                .expect("Error reading SOUND_SIZE from file!");
            if size != SOUND_SIZE {
                println!("Check SOUND_SIZE in input file.")
            }
        }
        _ => println!("The input file is empty!"),
    }

    // The second line of the input file is the number of symbols included in the file:
    let ns = lines_iter.next();
    let mut num_symbols: usize = 0;

    match ns {
        Some(i) => {
            num_symbols = i.parse::<usize>().unwrap();
        }
        _ => println!("Error reading number of symbols in file!"),
    }
    // println!("num_symbols is {}", num_symbols);

    // Now we can loop num_symbols times and store each symbol in a hashmap:
    //  key: the meaning of the symbol (as a string), e.g., "eat"
    //  value: the symbol itself as an array of 1s and 0s (like a sound frame)
    let mut symbols: HashMap<String, Symbol> = HashMap::new();

    for _ in 0..num_symbols {
        // Read in the header for the symbol (e.g. "eat"):
        let header = lines_iter.next().unwrap();
        // Read in the rows of the current symbol:
        let mut symbol: Vec<Vec<String>> = Vec::new();
        for _ in 0..SOUND_SIZE {
            let row = lines_iter.next().unwrap();
            let split: Vec<String> = row.split(" ").map(|s| s.to_owned()).collect();
            symbol.push(split);
        }
        // println!("H: {}", header);
        symbols.insert(header, Symbol { frame: symbol });
    }

    symbols
}

fn create_input_file(
    transpose: bool,
    file: &mut File,
    file_actions: &mut File,
    symbol_id: u8,
    symbols: &HashMap<String, Symbol>,
) -> std::io::Result<()> {
    /* Writes spike chains to a file. Each row represents one millisecond. */

    // Create a file for standard output:
    // remove_file("sample_input.txt")?;
    // let mut file = File::create("sample_input.txt")?;

    // Create a second file of the same length that stores the action ("" or "eat" etc.) for each frame:
    // remove_file("sample_input_actions.txt")?;
    // let mut file_actions = File::create("sample_input_actions.txt")?;
    // For use in creating random numbers:
    let mut rng = thread_rng();

    // Create the original sound frame, all pixels white (0):
    let mut sound = [[0; SOUND_SIZE]; SOUND_SIZE];
    // This vector of length NUM_FRAMES represents periods of quiet and noise:
    let soundscape = create_soundscape(symbol_id, symbols);
    // [(1, "eat"), ... 30 times]
    // We store the input data as an array in case the user wishes to transpose the data:
    let mut input_array: Vec<Vec<String>> = Vec::new();

    for frame_idx in 0..NUM_FRAMES {
        for r in 0..SOUND_SIZE {
            // See https://codeinreview.com/86/modifying-the-contents-of-an-array-of-vectors-in-rust/
            // for an explanation of this next line, especially the &mut bit:
            let row = &mut sound[r];
            // For each pixel, toggle its color at random:
            for c in 0..SOUND_SIZE {
                match row[c] {
                    0 => {
                        if rng.gen::<f64>() < PROB_TO_BLACK {
                            row[c] = 1;
                        }
                    }
                    1 => {
                        if rng.gen::<f64>() < PROB_TO_WHITE {
                            row[c] = 0;
                        }
                    }
                    _ => println!("There is an integer other than 0 or 1 in our frame!"),
                }
            }
        }

        // The ambient noise is complete! Now check if we are currently transmitting a symbol:

        if soundscape[frame_idx as usize].0 == 1 {
            // ..then overlay the symbol onto the ambient sound frame:

            let action = &soundscape[frame_idx as usize].1;
            let symbol = &symbols.get(action).unwrap().frame;
            // let symbol = &symbols.get(sym_id);

            // Now XOR the symbol's 1s with the ambient frame's 1s:
            for x in 0..SOUND_SIZE {
                for y in 0..SOUND_SIZE {
                    if sound[x][y] == 0 && symbol[x][y] == "1".to_string() {
                        sound[x][y] = 1;
                    }
                    // Finally, switch 0 <-> 1 at random to simulate noise:
                    if rng.gen::<f64>() < PROB_NOISE {
                        sound[x][y] = (sound[x][y] + 1) % 2;
                    }
                }
            }
        }

        // For debugging:
        // println!("\n{:?}", &soundscape[frame_idx as usize]);
        // for row in &sound {
        //     println!("{:?}", row);
        // }
        // println!("");

        if frame_idx == 0 {
            writeln!(
                file_actions,
                "{}",
                &soundscape[frame_idx as usize].1.to_string()
            )?;
        }

        /* A lengthy explanation of how we create the final input vector:
        Now we create the final input vector, which will contain only 1s and 0s. This is
        the vector that we will feed to the liquid state machine.

        For each pixel in the sound frame, we look at all pixels within distance d from
        the pixel, using the taxicab metric. When d = 0, we are looking at just the pixel
        itself. The maximum d is determined by considering the pixel in the center of
        the frame and finding the taxicab distance to the closest corner. (This requires
        us to assess whether the frame is odd x odd or even x even, but this is trivial.)

        For a given distance d from a pixel P, we check if the majority of the pixels
        within d of P are black (1). If so, then the final input vector has a 1 associated
        with that distance and pixel.

        10010  For example, suppose we have a 5x5 sound frame like this one.
        01101  The maximum d for a 5x5 frame is 4 (as explained above).
        00111  So suppose that P is the pixel in the center of the top row, and d = 3.
        10101  The pixels that we consider in this case are marked with X in the figure
        11000  below this one; each is within distance 3 of P.

        XXXXX  Counting the 1s and 0s covered by Xs, we count eight 1s and six 0s.
        XXXXX  Thus, this neighborhood of P is 'noisy' and, in the final input vector,
        0XXX1  we would use a 1 to denote this. We do not currently care how much
        10X01  noise there is -- only whether the neighborhood is noisy.
        11000

        The final input vector will contain the entries [0, 0, 1, 1, 0] for P once we
        are finished, where the jth entry is 0 if the pixels within distance j of P
        are not noisy, and 1 if they are noisy. This example is given as a check.
        */

        // Determine the maximum of all minimum taxicab distances from each pixel:
        let max_dist: usize = 0;

        // match SOUND_SIZE % 2 {
        //     0 => max_dist = SOUND_SIZE - 2, // if even, subtract 2
        //     _ => max_dist = SOUND_SIZE - 1, // if odd, subtract 1
        // }

        // Now consider each pixel and loop through all pixels (including itself),
        // creating a vector v = [0, 0, 0, 0] where the size of v is max_dist + 1,
        // and each v[k] equals 1 if the majority of pixels AT (not within) distance
        // k are black.

        let mut input: Vec<isize> = Vec::new();

        for i in 0..SOUND_SIZE {
            for j in 0..SOUND_SIZE {
                let mut v = vec![0; max_dist + 1];

                for x in 0..SOUND_SIZE {
                    for y in 0..SOUND_SIZE {
                        let td = get_taxicab(vec![i, j], vec![x, y]) as usize;

                        if td <= max_dist {
                            match &sound[x][y] {
                                0 => v[td] -= 1,
                                1 => v[td] += 1,
                                _ => println!("Eek! Check max_dist vs SOUND_SIZE.."),
                            }
                        }
                    }
                }

                // The vector v contains the appropriate entries for each specific taxicab
                // distance (from the current pixel). We want the partial sums of this vector:
                let mut v_partials = partial_sums(v);
                input.append(&mut v_partials);
            }
        }

        // println!("final unfiltered input is \n {:?}", input);

        let final_input = convert_to_01(input);
        // println!("final filtered input is \n {:?}", final_input);

        let strings: Vec<String> = final_input.iter().map(|n| n.to_string()).collect();

        // If the user wishes to transpose the final input, we store it as we go:
        if transpose {
            input_array.push(strings.clone());
        } else {
            // Write the final_input (as strings, for debugging) to the output file:
            writeln!(file, "{}", strings.join(" "))?;
        }
    }

    // Now transpose the final input data if requested:
    // let mut transposed_input: Vec<Vec<String>> = Vec::new();
    if transpose {
        for i in 0..input_array[0].len() {
            let mut new_row: Vec<String> = Vec::new();
            for j in 0..input_array.len() {
                let x = &input_array[j][i];
                new_row.push(x.to_string());
            }
            writeln!(file, "{}", new_row.join(" "))?;
        }
    }

    Ok(())
}

fn create_soundscape(sym_id: u8, symbols: &HashMap<String, Symbol>) -> Vec<(u8, String)> {
    /* Tuples in this vector will look like (0, "") or (1, "eat") where 0 represents
    a quiet millisecond and 1 represents a symbol is being transmitted, and the string
    tells us which symbol it is. */
    // let symbols = load_symbols();

    // Grab all the symbol meanings from the symbols HashMap:
    let mut meanings: Vec<String> = Vec::new();
    for (key, _) in symbols {
        meanings.push(key.clone());
    }

    // Let's start with a quiet millisecond:
    let mut soundscape: Vec<(u8, String)> = vec![/*(0, "nothing".to_string())*/];
    let mut quiet = false;
    if sym_id as usize % (meanings.len() + 1) == meanings.len() {
        quiet = true;
    }
    // println!("id: {}, quiet: {}", sym_id, quiet);

    let mut _rng = rand::thread_rng();

    for _ in 0..NUM_FRAMES {
        assert_eq!(NUM_FRAMES, BURST_MEAN as isize);
        if quiet {
            // let pv: f32 = generate_poisson_value();
            // let pv_int = pv as u8;
            for _ in 0..1 {
                soundscape.push((0, "nothing".to_string()));
            }
        // quiet = false;
        } else {
            // let nv: f32 = generate_normal_value();
            //let nv_int = BURST_MEAN as u8; //nv as u8;
            // sym_id: u8 = {
            // }; // rng.gen_range(0, meanings.len() as u8);
            // let _sym_id: u8 = _rng.gen_range(0, meanings.len() as u8);
            let m = &meanings[sym_id as usize];
            // for _ in 0..nv_int {
            // }
            soundscape.push((1, m.to_string()));
        }
    }

    // println!("{}, {:?}", sym_id, meanings);
    // println!("{}, {},\n {:?}\n\n", sym_id, soundscape.len(), soundscape);

    // Slice, because the loop above might allow for soundscapes longer than NUM_FRAMES:
    soundscape[..(NUM_FRAMES as usize)].to_vec()
}

fn generate_poisson_value() -> f32 {
    let poi = Poisson::new(POISSON_MEAN).unwrap();
    poi.sample(&mut rand::thread_rng())
}

fn generate_normal_value() -> f32 {
    let normal = Normal::new(BURST_MEAN, BURST_SD).unwrap();
    normal.sample(&mut rand::thread_rng())
}

fn get_taxicab(pt1: Vec<usize>, pt2: Vec<usize>) -> isize {
    /* Returns the taxicab distance between the two given points. */

    let a = pt1[0] as isize;
    let b = pt1[1] as isize;
    let c = pt2[0] as isize;
    let d = pt2[1] as isize;
    (b - d).abs() + (a - c).abs()
}

fn partial_sums(v: Vec<isize>) -> Vec<isize> {
    /* Returns a vector w with the property that w[k] is the sum of v[0] through v[k]. */

    let mut w: Vec<isize> = vec![0; v.len()];
    for i in 0..v.len() {
        let t: isize = v[..i + 1].iter().sum();
        w[i] = t;
    }
    w
}

fn convert_to_01(input: Vec<isize>) -> Vec<isize> {
    /* Convert all entries 1 or greater to 1s, and all entries 0 or less to 0s. */

    let mut final_input = vec![0; input.len()];
    for i in 0..input.len() {
        if input[i] > 0 {
            final_input[i] = 1;
        } else {
            final_input[i] = 0;
        }
    }
    final_input
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taxicab_check_01() {
        assert_eq!(2, get_taxicab(vec![0, 0], vec![1, 1]));
    }

    #[test]
    fn taxicab_check_02() {
        assert_eq!(3, get_taxicab(vec![0, 0], vec![2, 1]));
    }

    #[test]
    fn taxicab_check_03() {
        assert_eq!(0, get_taxicab(vec![4, 4], vec![4, 4]));
    }

    #[test]
    fn taxicab_check_04() {
        assert_ne!(2, get_taxicab(vec![0, 0], vec![2, 1]));
    }

    #[test]
    fn partial_sums_check_01() {
        assert_eq!(vec![0, 1, 3], partial_sums(vec![0, 1, 2]));
    }

    #[test]
    fn partial_sums_check_02() {
        assert_eq!(vec![2, 2, 2, 2], partial_sums(vec![2, 0, 0, 0]));
    }

    #[test]
    fn partial_sums_check_03() {
        assert_eq!(vec![-2, 0, -1, 2], partial_sums(vec![-2, 2, -1, 3]));
    }

    #[test]
    fn convert_to_01_check_01() {
        assert_eq!(vec![0, 1, 0, 1, 0], convert_to_01(vec![-2, 2, -1, 3, 0]));
    }

    #[test]
    fn convert_to_01_check_02() {
        assert_eq!(vec![1], convert_to_01(vec![9]));
    }
}

#[derive(Debug)]
struct Symbol {
    /* A Symbol is the same size as a sound frame, and carries a particular meaning. */
    frame: Vec<Vec<String>>,
}
