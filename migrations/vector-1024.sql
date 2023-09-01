CREATE TABLE document (
    id serial primary key,
    "text" text not null,
    source text not null,
    embedding vector(1024)
);
