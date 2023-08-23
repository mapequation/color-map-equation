use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::env::Args;
use std::error::Error;
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::str::FromStr;
use std::{env, fs, process};

use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

type Node = usize;
type Source = usize;
type Target = usize;
type Edge = (Source, Target);
type Weight = usize;

type CategoryMeta = usize;
type RealValueMeta = f64;

type Matrix = Vec<Vec<Weight>>;
type EdgeList = Vec<Edge>;


fn simulate_categorical(
    matrix: &Matrix,
    metadata: &[CategoryMeta],
    same_prob: f64,
    diff_prob: f64,
    n_samples: usize,
) -> Matrix {
    let mut rng = thread_rng();

    let out_links_for_node: Vec<Vec<Target>> = matrix.iter().map(flatnonzero).collect();

    let mut links: Matrix = matrix.iter().map(zeros_as).collect();

    for source in 0..matrix.len() {
        let source_meta = metadata[source];

        if out_links_for_node[source].len() == 0 {
            continue;
        }

        for _ in 0..n_samples {
            let mut target = source;

            let coded = loop {
                let out_links = &out_links_for_node[target];

                if out_links.len() == 0 {
                    break false;
                }

                target = *out_links.choose(&mut rng).unwrap();

                if source == target {
                    continue;
                }

                let same_meta = source_meta == metadata[target];

                let r: f64 = rng.gen();

                let code_same = r < same_prob;
                let code_diff = r < diff_prob;

                if (same_meta && code_same) || (!same_meta && code_diff) {
                    break true;
                }
            };

            if coded {
                links[source][target] += 1;
            }
        }
    }

    links
}

fn simulate_real(
    matrix: &Matrix,
    metadata: &[RealValueMeta],
    s: f64,
    meta_scale: f64,
    n_samples: usize,
) -> Matrix {
    let mut rng = thread_rng();

    let out_links_for_node: Vec<Vec<Target>> = matrix.iter().map(flatnonzero).collect();

    let mut links: Matrix = matrix.iter().map(zeros_as).collect();
    
    const MAX_ATTEMPTS: usize = 1_000;
    
    for source in 0..matrix.len() {
        let source_meta = metadata[source];

        if out_links_for_node[source].len() == 0 {
            continue;
        }

        for _ in 0..n_samples {
            let mut target = source;
            let mut attempts = 0;
            
            let coded = loop {

                let out_links = &out_links_for_node[target];

                if out_links.len() == 0 {
                    break false;
                }

                target = *out_links.choose(&mut rng).unwrap();

                if source == target {
                    attempts += 1;
                    
                    if attempts > 2 * out_links.len() {
                        break false;
                    }
                    
                    continue;
                }

                let target_meta = metadata[target];

                let code_prob = (1.0 - s)
                    * f64::powf(
                        std::f64::consts::E,
                        -(source_meta - target_meta).abs() / meta_scale,
                    ) + s;

                let r: f64 = rng.gen();

                if r < code_prob {
                    break true;
                }
                
                attempts += 1;
                
                if attempts > MAX_ATTEMPTS {
                    break false;
                }
            };

            if coded {
                links[source][target] += 1;
            }
        }
    }

    links
}

fn zeros_as<T, U>(a: &Vec<U>) -> Vec<T>
where
    T: Default + Clone,
{
    vec![T::default(); a.len()]
}

// Similar to np.flatnonzero
fn flatnonzero(a: &Vec<usize>) -> Vec<usize> {
    a.iter()
        .enumerate()
        .filter(|(_, &elem)| elem > 0)
        .map(|(idx, _)| idx)
        .collect()
}

fn run(config: Config) -> Result<(), Box<dyn Error>> {
    let Config {
        input_file,
        meta_file,
        metadata_type,
        out_file,
        same_prob,
        diff_prob,
        meta_scale,
        n_samples,
    } = config;

    let matrix = parse_edgelist(&input_file).unwrap();

    let links = match metadata_type {
        Metadata::Categorical => {
            let meta = parse_metadata(&meta_file).unwrap();
            simulate_categorical(&matrix, &meta, same_prob, diff_prob, n_samples)
        }
        Metadata::RealValued => {
            let meta = parse_metadata(&meta_file).unwrap();
            let s = same_prob;
            simulate_real(&matrix, &meta, s, meta_scale, n_samples)
        }
    };

    let degree: Vec<usize> = matrix
        .iter()
        .map(|row| row.iter().map(|&w| if w > 0 { 1 } else { 0 }).sum())
        .collect();

    let tot_degree: usize = degree.iter().sum();

    let mut f = BufWriter::new(File::create(&out_file)?);

    for (source, row) in links.iter().enumerate() {
        for (target, &weight) in row.iter().enumerate() {
            if weight > 0 {
                let flow = weight as f64 / n_samples as f64;
                let degree = degree[source] as f64 / tot_degree as f64;

                writeln!(f, "{} {} {}", source, target, flow * degree)?;
            }
        }
    }

    Ok(())
}

fn main() {
    let config = env::args().try_into().unwrap_or_else(|err| {
        eprintln!("{}", err);
        process::exit(1);
    });

    if let Err(err) = run(config) {
        eprintln!("{}", err);
        process::exit(1);
    }
}

fn parse_edgelist(lines: &str) -> Result<Matrix, Box<dyn Error>> {
    let mut max_id = 0;

    let edges: EdgeList = lines
        .lines()
        .map(|line| {
            let edge = line
                .split_whitespace()
                .map(|node| node.parse().unwrap())
                .collect::<Vec<Node>>();

            debug_assert_eq!(edge.len(), 2);

            max_id = max_id.max(edge[0]);
            max_id = max_id.max(edge[1]);

            (edge[0], edge[1])
        })
        .collect();

    let mut matrix = vec![vec![0; max_id + 1]; max_id + 1];

    for &(source, target) in edges.iter() {
        matrix[source][target] = 1;
    }

    Ok(matrix)
}

fn parse_metadata<T>(contents: &str) -> Result<Vec<T>, Box<dyn Error>>
where
    T: FromStr + Copy + Clone + Default,
    <T as FromStr>::Err: Debug,
{
    let mut metadata_map: HashMap<Node, T> = HashMap::new();
    let mut max_id = 0;

    for line in contents.lines() {
        let line: Vec<&str> = line.split_whitespace().collect();

        assert_eq!(line.len(), 2);

        let node = line[0].parse().unwrap();
        let meta = line[1]
            .parse()
            .expect(&format!("Could not parse metadata {:?}", line));

        max_id = max_id.max(node);

        metadata_map.insert(node, meta);
    }

    let mut metadata = vec![T::default(); max_id + 1];

    for (&node, &meta) in metadata_map.iter() {
        metadata[node] = meta;
    }

    Ok(metadata)
}

enum Metadata {
    Categorical,
    RealValued,
}

struct Config {
    pub input_file: String,
    pub meta_file: String,
    pub metadata_type: Metadata,
    pub out_file: String,
    pub same_prob: f64,
    pub diff_prob: f64,
    pub meta_scale: f64,
    pub n_samples: usize,
}

impl TryFrom<std::env::Args> for Config {
    type Error = Box<dyn Error>;

    fn try_from(mut args: Args) -> Result<Self, Self::Error> {
        let prog_name = args.next().unwrap();

        if args.len() < 6 {
            return Err(format!(
                "Usage:\n\
                {} -c input_file meta_file out_file same_prob diff_prob n_samples \n\
                {} -r input_file meta_file out_file s meta_scale n_samples",
                prog_name, prog_name
            )
            .into());
        }

        let metadata_type = match args.next() {
            None => return Err("Must specify -c (categorical) or -r (real valued) metadata".into()),
            Some(arg) => match arg.as_str() {
                "-c" => Metadata::Categorical,
                "-r" => Metadata::RealValued,
                _ => {
                    return Err("Must specify -c (categorical) or -r (real valued) metadata".into())
                }
            },
        };

        let input_file = match args.next() {
            None => return Err("Missing input filename".into()),
            Some(arg) => fs::read_to_string(arg).expect("Cannot open file"),
        };

        let meta_file = match args.next() {
            None => return Err("Missing meta filename".into()),
            Some(arg) => fs::read_to_string(arg).expect("Cannot open file"),
        };

        let out_file = args.next().unwrap();
        let same_prob = args.next().unwrap().parse()?;

        let (diff_prob, meta_scale) = {
            let next_arg = args.next().unwrap().parse()?;

            match metadata_type {
                Metadata::Categorical => (next_arg, 0.0),
                Metadata::RealValued => (0.0, next_arg),
            }
        };

        let n_samples = args.next().unwrap().parse()?;

        Ok(Self {
            input_file,
            meta_file,
            metadata_type,
            out_file,
            same_prob,
            diff_prob,
            meta_scale,
            n_samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate() {
        let matrix: Matrix = vec![
            vec![0, 1, 0, 1],
            vec![1, 0, 0, 0],
            vec![0, 0, 0, 0],
            vec![1, 0, 0, 0],
        ];

        let color: [usize; 4] = [1, 1, 1, 1];

        let same_prob = 1.0;
        let diff_prob = 1.0 * same_prob;

        let links = simulate_categorical(&matrix, &color, same_prob, diff_prob, 100_000);

        println!("{:?}", links);
    }

    #[test]
    fn test_parse_edgelist() {
        let lines = "1 2\n3 4\n2 1\n";

        let edges = parse_edgelist(lines).unwrap();

        assert_eq!(
            edges,
            vec![
                vec![0, 0, 0, 0, 0],
                vec![0, 0, 1, 0, 0],
                vec![0, 1, 0, 0, 0],
                vec![0, 0, 0, 0, 1],
                vec![0, 0, 0, 0, 0]
            ]
        );
    }

    #[test]
    fn test_parse_categories() {
        let meta = "1 1\n2 1\n3 2\n4 1\n";

        let meta = parse_metadata::<usize>(meta).unwrap();

        assert_eq!(meta, vec![0, 1, 1, 2, 1]);
    }

    #[test]
    fn test_parse_real_valued() {
        let meta = "1 0.0\n2 0.1\n3 3.1314\n4 2.74\n";

        let meta = parse_metadata::<f64>(meta).unwrap();

        assert_eq!(meta, vec![0.0, 0.0, 0.1, 3.1314, 2.74])
    }
}
