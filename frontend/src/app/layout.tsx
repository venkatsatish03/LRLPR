import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "License Plate Recognition",
  description: "Upload a vehicle image and detect the license plate.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
