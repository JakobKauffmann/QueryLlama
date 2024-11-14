import pandas as pd
from langchain_community.chat_models import ChatOllama
import click

@click.group()
def cli():
    pass

@cli.command()
@click.option(
    "--queries-path",
        type=click.Path(exists=True),
        required=True,
        help="The path to the CSV file containing queries."
)
@click.option(
    "--output-directory",
    required=True,
    help="The path to the directory in which to save batches of responses."
)
@click.option(
    "--batch-size",
    type=int,
    required=True,
    help="Number of queries to process and responses to save during each batch."
)
@click.option(
    "--model",
    required=False,
    default="llama3.2:3b",
    help="Llama model to use."
)
@click.option(
    "--url",
    required=False,
    default="http://localhost:11434/api/chat",
    help="Ensure the local server is running at this url"
)
def query_llama(queries_path: str, output_directory:str, batch_size:int, model: str, url: str):

    # Load the CSV
    queries_df = pd.read_csv(queries_path)

    # Check for the 'query' column in the DataFrame
    if "query" not in queries_df.columns:
        raise ValueError("The CSV file must have a 'query' column.")

    queries = queries_df["query"].tolist()

    # Initialize the Llama model
    llm = ChatOllama(model="llama3.2:3b")

    # Process in Batches of user input batch_size

    num_batches = len(queries) // batch_size + (1 if len(queries) % batch_size != 0 else 0)

    for batch_num in range(125, num_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, len(queries))
        batch_queries = queries[batch_start:batch_end]

        print(f"Processing batch {batch_num + 1}/{num_batches} with {len(batch_queries)} queries.")

        batch_responses = []
        for cnt, prompt in enumerate(batch_queries):
            response = llm.invoke(prompt)
            batch_responses.append({"prompt": prompt, "response": response.content})
            batch_progress = ((cnt + 1) / batch_size)*100
            print(f"\nProcessed prompt: {prompt}")
            print(f"\n {batch_progress}% done with batch {batch_num+1}")

        # Convert batch responses to a DataFrame and save to CSV
        batch_df = pd.DataFrame(batch_responses)
        batch_output_path = f"{output_directory}/llama_batch_{batch_num + 1}.csv"
        batch_df.to_csv(batch_output_path, index=False)

        print(f"Saved batch {batch_num + 1} responses to {batch_output_path}")

    # Optional: Merge all batches into a single CSV file after processing all batches
    all_batches = [pd.read_csv(f"{output_directory}/llama_batch_{i + 1}.csv") for i in range(num_batches)]
    final_df = pd.concat(all_batches, ignore_index=True)
    final_df.to_csv(f"{output_directory}/all_llama_responses.csv", index=False)
    print(f"All batches combined and saved to {output_directory}/all_llama_responses.csv")

def main():
    cli()


if __name__ == "__main__":
    main()
