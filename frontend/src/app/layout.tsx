import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LPR Investigation Dashboard",
  description: "Vehicle plate evidence review dashboard.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}

